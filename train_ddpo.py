#!/usr/bin/env python3
"""
train_ddpo.py — Main training script for DDPO baseline.

Usage:
    python train_ddpo.py --config configs/baseline.yaml
    python train_ddpo.py --config configs/baseline_a100.yaml --output_dir /content/drive/MyDrive/ddpo_baseline/run1
    python train_ddpo.py --config configs/baseline.yaml --resume_from /path/to/checkpoint/step_00100
    python train_ddpo.py --config configs/baseline.yaml --training.num_train_steps 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler, StableDiffusionPipeline
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm

from ddpo.diversity import CLIPDiversity, LPIPSDiversity
from ddpo.eval import evaluate, generate_before_after_grid
from ddpo.prompts import EVAL_PROMPTS, sample_train_prompts
from ddpo.rewards import ImageRewardScore, PickScoreReward
from ddpo.training import TrainConfig, train_one_iteration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ddpo")


# ── Config helpers ───────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply dot-notation CLI overrides like --training.lr 1e-4."""
    for i in range(0, len(overrides), 2):
        key_path = overrides[i].lstrip("-").split(".")
        value = overrides[i + 1]
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"

        d = cfg
        for k in key_path[:-1]:
            d = d.setdefault(k, {})
        d[key_path[-1]] = value
    return cfg


def _resolve_dtype(cfg: Dict[str, Any]) -> torch.dtype:
    """Pick frozen-component dtype from config (default bfloat16 on A100)."""
    name = cfg.get("model", {}).get("dtype", "bfloat16")
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


# ── Checkpoint save / load ───────────────────────────────────────────────────

def save_checkpoint(
    pipe: StableDiffusionPipeline,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics_history: list,
    prompt_rng: random.Random,
    save_dir: str,
) -> str:
    """Save full resumable checkpoint: LoRA weights + optimizer + RNG + history."""
    ckpt_path = os.path.join(save_dir, f"step_{step:05d}")
    os.makedirs(ckpt_path, exist_ok=True)

    pipe.unet.save_pretrained(os.path.join(ckpt_path, "lora"))

    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "prompt_rng_state": prompt_rng.getstate(),
    }, os.path.join(ckpt_path, "training_state.pt"))

    with open(os.path.join(ckpt_path, "metrics_history.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    logger.info("Saved checkpoint to %s", ckpt_path)
    return ckpt_path


def load_checkpoint(
    optimizer: torch.optim.Optimizer,
    resume_path: str,
    prompt_rng: random.Random,
) -> tuple[int, list]:
    """
    Restore optimizer state, RNG states, and metrics history from checkpoint.

    LoRA weights are loaded separately in setup_pipeline() via PeftModel.from_pretrained,
    so this function only handles the non-model state.

    Returns (step, metrics_history).
    """
    logger.info("Loading training state from: %s", resume_path)

    state_path = os.path.join(resume_path, "training_state.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=False)

    optimizer.load_state_dict(state["optimizer_state_dict"])

    torch.random.set_rng_state(state["torch_rng_state"])
    if state["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    prompt_rng.setstate(state["prompt_rng_state"])

    step = state["step"]

    history_path = os.path.join(resume_path, "metrics_history.json")
    with open(history_path) as f:
        metrics_history = json.load(f)

    logger.info("Resumed at step %d with %d history entries", step, len(metrics_history))
    return step, metrics_history


def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Find the most recent step_XXXXX checkpoint in save_dir, or None."""
    if not os.path.isdir(save_dir):
        return None
    subdirs = [
        d for d in os.listdir(save_dir)
        if d.startswith("step_") and os.path.isdir(os.path.join(save_dir, d))
    ]
    if not subdirs:
        return None
    latest = sorted(subdirs)[-1]
    full = os.path.join(save_dir, latest)
    state_file = os.path.join(full, "training_state.pt")
    if not os.path.isfile(state_file):
        return None
    return full


# ── Model setup ──────────────────────────────────────────────────────────────

def setup_pipeline(
    cfg: Dict[str, Any],
    device: str = "cuda",
    resume_path: Optional[str] = None,
) -> StableDiffusionPipeline:
    """Load SD 1.5, attach LoRA adapters, configure for DDPO training."""
    model_cfg = cfg["model"]
    frozen_dtype = _resolve_dtype(cfg)
    logger.info("Loading pipeline: %s (frozen dtype: %s)", model_cfg["sd_model_id"], frozen_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_cfg["sd_model_id"],
        torch_dtype=frozen_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Freeze everything
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # VAE in float32 for stability (even with bf16 base, VAE decode is safer in fp32)
    pipe.vae = pipe.vae.to(dtype=torch.float32)

    if resume_path is not None:
        lora_path = os.path.join(resume_path, "lora")
        if not os.path.isdir(lora_path):
            lora_path = resume_path
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=model_cfg["lora_rank"],
            lora_alpha=model_cfg["lora_alpha"],
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)

    pipe.unet.print_trainable_parameters()

    # LoRA params in fp32 for stable optimisation
    for name, param in pipe.unet.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # Gradient checkpointing AFTER peft wrapping (order matters)
    pipe.unet.base_model.enable_gradient_checkpointing()

    pipe.unet.eval()
    return pipe


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DDPO Baseline Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output dir (for Google Drive, etc.)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint dir to resume from, or 'auto' "
                             "to find the latest in the checkpoint save_dir.")
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config)
    if unknown:
        cfg = apply_overrides(cfg, unknown)

    # CLI --output_dir overrides both output and checkpoint dirs
    if args.output_dir is not None:
        cfg["logging"]["output_dir"] = args.output_dir
        cfg["checkpoint"]["save_dir"] = os.path.join(args.output_dir, "checkpoints")

    output_dir = cfg["logging"]["output_dir"]
    ckpt_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resolve resume path
    resume_path = args.resume_from
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(ckpt_dir)
        if resume_path:
            logger.info("Auto-resume: found %s", resume_path)
        else:
            logger.info("Auto-resume: no checkpoint found, starting fresh")

    # Seed (will be overwritten if resuming)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    frozen_dtype = _resolve_dtype(cfg)

    # ── Setup ────────────────────────────────────────────────────────────
    pipe = setup_pipeline(cfg, device, resume_path=resume_path)

    reward_model = PickScoreReward(device=device, dtype=frozen_dtype)
    diversity_fn = LPIPSDiversity(device=device)
    clip_diversity_fn = CLIPDiversity(device=device)
    
    try:
        image_reward_model = ImageRewardScore(device=device)
    except Exception as e:
        logger.warning(f"Failed to load ImageReward model: {e}. Evaluation will skip this metric.")
        image_reward_model = None

    train_cfg = TrainConfig(
        num_train_steps=cfg["training"]["num_train_steps"],
        num_prompts_per_iter=cfg["training"]["num_prompts_per_iter"],
        samples_per_prompt=cfg["training"]["samples_per_prompt"],
        inner_epochs=cfg["training"]["inner_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
        clip_range=cfg["training"]["clip_range"],
        advantage_clip=cfg["training"]["advantage_clip"],
        max_grad_norm=cfg["training"].get("max_grad_norm", 1.0),
        grad_accumulation_steps=cfg["training"].get("grad_accumulation_steps", 1),
        num_inference_steps=cfg["sampling"]["num_inference_steps"],
        ddim_eta=cfg["sampling"]["ddim_eta"],
        guidance_scale=cfg["sampling"]["guidance_scale"],
        image_size=cfg["sampling"]["image_size"],
    )

    # Optimizer: only LoRA params
    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    prompt_rng = random.Random(args.seed)
    all_metrics_history: list = []
    start_step = 0

    # ── Resume ───────────────────────────────────────────────────────────
    if resume_path is not None:
        start_step, all_metrics_history = load_checkpoint(
            optimizer, resume_path, prompt_rng,
        )
        logger.info("Resuming training from step %d", start_step)

    # ── Wandb ────────────────────────────────────────────────────────────
    use_wandb = cfg["logging"].get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            config=cfg,
            name=f"ddpo-baseline-{time.strftime('%Y%m%d-%H%M%S')}",
            resume="allow",
        )

    eval_every = cfg["evaluation"]["eval_every"]
    n_eval_samples = cfg["evaluation"]["n_samples_per_eval_prompt"]
    n_eval_intrain = cfg["evaluation"]["n_samples_per_eval_prompt_intrain"]

    # ── Initial eval (only if starting fresh) ────────────────────────────
    if start_step == 0:
        logger.info("Running initial evaluation (step 0)...")
        eval_metrics = evaluate(
            pipe, reward_model.score, diversity_fn,
            n_samples=n_eval_intrain,
            num_inference_steps=train_cfg.num_inference_steps,
            ddim_eta=train_cfg.ddim_eta,
            guidance_scale=train_cfg.guidance_scale,
            image_size=train_cfg.image_size,
            save_dir=os.path.join(output_dir, "eval_images"),
            step=0,
            clip_diversity_fn=clip_diversity_fn,
            image_reward_fn=image_reward_model.score if image_reward_model is not None else None,
        )
        logger.info("Step 0 eval: %s", json.dumps(eval_metrics, indent=2))
        all_metrics_history.append({"step": 0, **eval_metrics})

        if use_wandb:
            import wandb
            wandb.log(eval_metrics, step=0)

    # ── Training loop ────────────────────────────────────────────────────
    for step in tqdm(
        range(start_step + 1, train_cfg.num_train_steps + 1),
        initial=start_step,
        total=train_cfg.num_train_steps,
        desc="DDPO Training",
    ):
        base_prompts = sample_train_prompts(
            train_cfg.num_prompts_per_iter, rng=prompt_rng
        )
        prompts = []
        for p in base_prompts:
            prompts.extend([p] * train_cfg.samples_per_prompt)

        gen = torch.Generator(device=device).manual_seed(args.seed + step)

        metrics = train_one_iteration(
            pipe, reward_model.score, optimizer, train_cfg,
            prompts, step, generator=gen,
        )

        if step % cfg["logging"].get("log_every", 1) == 0:
            logger.info("Step %d: %s", step, {k: f"{v:.4f}" for k, v in metrics.items()})

        if use_wandb:
            import wandb
            wandb.log(metrics, step=step)

        # ── Eval + Checkpoint ────────────────────────────────────────────
        if step % eval_every == 0 or step == train_cfg.num_train_steps:
            is_final = step == train_cfg.num_train_steps
            ns = n_eval_samples if is_final else n_eval_intrain

            logger.info("Evaluating at step %d (n_samples=%d)...", step, ns)
            eval_metrics = evaluate(
                pipe, reward_model.score, diversity_fn,
                n_samples=ns,
                num_inference_steps=train_cfg.num_inference_steps,
                ddim_eta=train_cfg.ddim_eta,
                guidance_scale=train_cfg.guidance_scale,
                image_size=train_cfg.image_size,
                save_dir=os.path.join(output_dir, "eval_images"),
                step=step,
                clip_diversity_fn=clip_diversity_fn,
                image_reward_fn=image_reward_model.score if image_reward_model is not None else None,
            )
            eval_metrics["step"] = step
            all_metrics_history.append(eval_metrics)
            logger.info("Step %d eval: %s", step, json.dumps(
                {k: f"{v:.4f}" if isinstance(v, float) else v
                 for k, v in eval_metrics.items()},
                indent=2,
            ))

            if use_wandb:
                import wandb
                wandb.log(eval_metrics, step=step)

            if cfg["checkpoint"].get("save_every_eval", True):
                save_checkpoint(
                    pipe, optimizer, step,
                    all_metrics_history, prompt_rng, ckpt_dir,
                )

    # ── Final outputs ────────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, "metrics_history.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics_history, f, indent=2)
    logger.info("Saved metrics history to %s", metrics_path)

    example_prompts = [
        "a single red apple on a white table",
        "a dog playing fetch in a sunny park",
        "a dreamlike landscape",
    ]
    example_categories = ["constrained", "common_subject", "open_ended"]
    example_seeds = [0, 1, 2, 3, 4, 5]

    logger.info("Generating before/after comparison grid...")
    generate_before_after_grid(
        pipe, example_prompts, example_categories, example_seeds,
        num_inference_steps=train_cfg.num_inference_steps,
        ddim_eta=train_cfg.ddim_eta,
        guidance_scale=train_cfg.guidance_scale,
        image_size=train_cfg.image_size,
        save_path=os.path.join(output_dir, "before_after_grid.png"),
    )

    if use_wandb:
        import wandb
        wandb.log({"before_after_grid": wandb.Image(
            os.path.join(output_dir, "before_after_grid.png")
        )})
        wandb.finish()

    logger.info("Training complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
