"""
Evaluation harness for DDPO.

Generates images for held-out eval prompts, computes per-prompt and aggregate
PickScore + LPIPS diversity, and optionally produces before/after comparison
grids.

IMPORTANT: Eval uses the SAME sampling settings as training (guidance_scale=1.0,
η=1.0, custom DDIM loop) so metrics reflect the actual policy.  We do NOT call
pipe(prompt) which applies CFG and different defaults.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from .diversity import LPIPSDiversity
from .prompts import EVAL_PROMPTS
from .sampling import decode_latents, encode_prompt, rollout

logger = logging.getLogger(__name__)


def evaluate(
    pipe: StableDiffusionPipeline,
    reward_fn,
    diversity_fn: LPIPSDiversity,
    n_samples: int = 50,
    num_inference_steps: int = 50,
    ddim_eta: float = 1.0,
    guidance_scale: float = 1.0,
    image_size: int = 512,
    save_dir: Optional[str] = None,
    step: int = 0,
    clip_diversity_fn=None,
    image_reward_fn=None,
) -> Dict[str, float]:
    """
    Run evaluation on the held-out prompt set.

    For each eval prompt, generates ``n_samples`` images with different seeds
    (same prompt, varied noise).  Computes:
      - PickScore (mean over n_samples)
      - LPIPS diversity (mean pairwise distance among n_samples)
      - CLIP embedding variance (if clip_diversity_fn provided)
      - ImageReward (if image_reward_fn provided)

    Args:
        pipe:               SD pipeline (LoRA enabled or disabled).
        reward_fn:          callable(images, prompts) → (B,) scores.
        diversity_fn:       LPIPSDiversity instance.
        n_samples:          images per eval prompt.
        num_inference_steps: DDIM steps.
        ddim_eta:           stochasticity.
        guidance_scale:     CFG weight (1.0 for no CFG).
        image_size:         spatial resolution.
        save_dir:           if set, save sample grids here.
        step:               training step (for filenames).
        clip_diversity_fn:  optional CLIPDiversity instance.
        image_reward_fn:    optional ImageRewardScore.score callable.

    Returns:
        Dict of aggregated metrics.
    """
    device = next(pipe.unet.parameters()).device
    pipe.unet.eval()

    metric_keys = ["pickscore", "diversity"]
    if clip_diversity_fn is not None:
        metric_keys.append("clip_variance")
    if image_reward_fn is not None:
        metric_keys.append("image_reward")

    results: Dict[str, Dict[str, List[float]]] = {
        cat: {k: [] for k in metric_keys}
        for cat in EVAL_PROMPTS
    }

    for category, prompts in EVAL_PROMPTS.items():
        for prompt in prompts:
            images = _generate_n_images(
                pipe, prompt, n_samples, num_inference_steps,
                ddim_eta, guidance_scale, image_size, device,
            )

            # PickScore
            scores = reward_fn(images, [prompt] * n_samples)
            results[category]["pickscore"].append(scores.mean().item())

            # LPIPS diversity (intra-prompt)
            div = diversity_fn.compute(images)
            results[category]["diversity"].append(div)

            # CLIP embedding variance
            if clip_diversity_fn is not None:
                clip_var = clip_diversity_fn.compute(images)
                results[category]["clip_variance"].append(clip_var)

            # ImageReward
            if image_reward_fn is not None:
                ir_scores = image_reward_fn(images, [prompt] * n_samples)
                results[category]["image_reward"].append(ir_scores.mean().item())

            if save_dir is not None:
                _save_grid(
                    images[:8], prompt, category,
                    os.path.join(save_dir, f"step_{step:05d}"),
                )

    # ── Aggregate ────────────────────────────────────────────────────────
    metrics: Dict[str, float] = {}
    all_values: Dict[str, list] = {k: [] for k in metric_keys}

    for cat in EVAL_PROMPTS:
        for k in metric_keys:
            vals = results[cat][k]
            metrics[f"eval/{cat}/{k}"] = np.mean(vals)
            all_values[k].extend(vals)

    for k in metric_keys:
        metrics[f"eval/overall/{k}"] = np.mean(all_values[k])

    return metrics


def _generate_n_images(
    pipe: StableDiffusionPipeline,
    prompt: str,
    n: int,
    num_inference_steps: int,
    ddim_eta: float,
    guidance_scale: float,
    image_size: int,
    device: torch.device,
) -> List[Image.Image]:
    """Generate n images for one prompt with different seeds."""
    all_images: List[Image.Image] = []
    batch_size = min(n, 4)  # avoid OOM on eval

    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        prompts_batch = [prompt] * bs
        prompt_embeds = encode_prompt(pipe, prompts_batch).to(device)

        gen = torch.Generator(device=device).manual_seed(start)
        data = rollout(
            pipe, prompt_embeds,
            num_inference_steps=num_inference_steps,
            eta=ddim_eta,
            guidance_scale=guidance_scale,
            generator=gen,
            image_size=image_size,
        )
        final_latents = data.latents[-1].to(device)
        images = decode_latents(pipe, final_latents)
        all_images.extend(images)

    return all_images[:n]


def _save_grid(
    images: List[Image.Image],
    prompt: str,
    category: str,
    save_dir: str,
) -> None:
    """Save a row of images as a grid."""
    os.makedirs(save_dir, exist_ok=True)
    safe_name = prompt[:60].replace(" ", "_").replace("/", "_")
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(f"[{category}] {prompt[:80]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=100)
    plt.close(fig)


# ── Before/after comparison grid ─────────────────────────────────────────────

def generate_before_after_grid(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    categories: List[str],
    seeds: List[int],
    num_inference_steps: int = 50,
    ddim_eta: float = 1.0,
    guidance_scale: float = 1.0,
    image_size: int = 512,
    save_path: str = "before_after.png",
) -> None:
    """
    Generate a comparison grid: pre-training vs post-training.

    Uses peft's disable_adapters()/enable_adapters() to toggle LoRA.
    Same seeds, same prompt, same scheduler config.

    Layout: for each prompt, row 0 = before (LoRA off), row 1 = after (LoRA on).
    """
    device = next(pipe.unet.parameters()).device
    n_prompts = len(prompts)
    n_seeds = len(seeds)

    fig, axes = plt.subplots(
        n_prompts * 2, n_seeds,
        figsize=(3 * n_seeds, 3 * n_prompts * 2),
    )
    if n_prompts * 2 == 1:
        axes = axes[np.newaxis, :]
    if n_seeds == 1:
        axes = axes[:, np.newaxis]

    for p_idx, (prompt, category) in enumerate(zip(prompts, categories)):
        for lora_on, label in [(False, "Before (base)"), (True, "After (LoRA)")]:
            row = p_idx * 2 + (1 if lora_on else 0)

            if not lora_on:
                if hasattr(pipe.unet, "disable_adapter_layers"):
                    pipe.unet.disable_adapter_layers()
                else:
                    pipe.unet.disable_adapters()

            for s_idx, seed in enumerate(seeds):
                prompt_embeds = encode_prompt(pipe, [prompt]).to(device)
                gen = torch.Generator(device=device).manual_seed(seed)

                with torch.no_grad():
                    data = rollout(
                        pipe, prompt_embeds,
                        num_inference_steps=num_inference_steps,
                        eta=ddim_eta,
                        guidance_scale=guidance_scale,
                        generator=gen,
                        image_size=image_size,
                    )
                final_latents = data.latents[-1].to(device)
                imgs = decode_latents(pipe, final_latents)

                axes[row, s_idx].imshow(imgs[0])
                axes[row, s_idx].axis("off")
                if s_idx == 0:
                    axes[row, s_idx].set_ylabel(
                        f"{label}\n[{category}]", fontsize=8, rotation=0,
                        labelpad=80, va="center",
                    )

            if not lora_on:
                if hasattr(pipe.unet, "enable_adapter_layers"):
                    pipe.unet.enable_adapter_layers()
                else:
                    pipe.unet.enable_adapters()

            axes[p_idx * 2, n_seeds // 2].set_title(
                prompt[:60], fontsize=8, pad=10,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved before/after grid to %s", save_path)
