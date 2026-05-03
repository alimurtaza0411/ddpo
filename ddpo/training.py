"""
PPO-style policy-gradient update loop for DDPO.

Per-iteration flow:
  1. Rollout: sample trajectories (no grad)
  2. Decode + score with PickScore
  3. Normalise advantages
  4. Inner PPO loop: for each epoch, shuffle timesteps, recompute log-probs
     under current policy, apply clipped surrogate loss, backprop, step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

from .sampling import (
    RolloutData,
    compute_log_prob_for_step,
    decode_latents,
    encode_prompt,
    rollout,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyper-parameters (mirroring the YAML structure)."""

    num_train_steps: int = 500
    num_prompts_per_iter: int = 8
    samples_per_prompt: int = 4
    inner_epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    clip_range: float = 0.1
    advantage_clip: float = 5.0
    max_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1
    num_inference_steps: int = 50
    ddim_eta: float = 1.0
    guidance_scale: float = 1.0
    image_size: int = 512


def compute_advantages(
    rewards: torch.Tensor,
    clip_value: float = 5.0,
) -> torch.Tensor:
    """
    Normalise rewards → zero-mean, unit-variance advantages, clamped.

    Args:
        rewards:    (B,) raw reward scores.
        clip_value: clamp magnitude.

    Returns:
        advantages: (B,) normalised & clamped.
    """
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return adv.clamp(-clip_value, clip_value)


def ppo_step(
    pipe: StableDiffusionPipeline,
    rollout_data: RolloutData,
    advantages: torch.Tensor,
    old_log_probs: List[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    global_step: int,
) -> Dict[str, float]:
    """
    One inner PPO update pass over all timesteps.

    Iterates timesteps in shuffled order.  For each timestep:
      - Move stored latents to GPU
      - Forward UNet (with grad) to get new log-prob
      - Compute clipped PPO loss
      - Backward + (optional) gradient accumulation
      - Step optimizer

    Returns dict of scalar metrics for logging.
    """
    device = next(pipe.unet.parameters()).device
    prompt_embeds = rollout_data.prompt_embeds.to(device)

    timesteps = list(range(len(rollout_data.timesteps)))
    perm = torch.randperm(len(timesteps)).tolist()

    total_loss = 0.0
    total_ratio = 0.0
    total_clipped_frac = 0.0
    n_steps = 0

    pipe.unet.train()
    optimizer.zero_grad()

    for idx_in_perm, step_idx in enumerate(perm):
        t = rollout_data.timesteps[step_idx]

        x_t = rollout_data.latents[step_idx].to(device).detach()
        x_prev = rollout_data.latents[step_idx + 1].to(device).detach()
        old_lp = old_log_probs[step_idx].to(device)

        new_lp = compute_log_prob_for_step(
            pipe,
            x_t,
            x_prev,
            t,
            prompt_embeds,
            eta=cfg.ddim_eta,
            guidance_scale=cfg.guidance_scale,
        )

        # PPO clipped surrogate
        log_ratio = new_lp - old_lp
        # Clamp log_ratio to avoid exp() overflowing into Inf
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)
        
        adv = advantages.to(device)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
        loss = -torch.min(surr1, surr2).mean()

        # Scale loss for gradient accumulation
        scaled_loss = loss / cfg.grad_accumulation_steps
        scaled_loss.backward()

        total_loss += loss.item()
        total_ratio += ratio.mean().item()
        total_clipped_frac += (
            (ratio < 1.0 - cfg.clip_range) | (ratio > 1.0 + cfg.clip_range)
        ).float().mean().item()
        n_steps += 1

        # Gradient accumulation: step every grad_accumulation_steps OR at end
        if (idx_in_perm + 1) % cfg.grad_accumulation_steps == 0 or (
            idx_in_perm + 1
        ) == len(perm):
            nn.utils.clip_grad_norm_(
                [p for p in pipe.unet.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

    pipe.unet.eval()

    return {
        "loss": total_loss / max(n_steps, 1),
        "mean_ratio": total_ratio / max(n_steps, 1),
        "clipped_frac": total_clipped_frac / max(n_steps, 1),
    }


def train_one_iteration(
    pipe: StableDiffusionPipeline,
    reward_fn,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    prompts: List[str],
    global_step: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """
    Full DDPO iteration: rollout → score → advantage → PPO update.

    Args:
        pipe:        SD pipeline with LoRA.
        reward_fn:   callable(images, prompts) → (B,) reward tensor.
        optimizer:   AdamW on LoRA params.
        cfg:         TrainConfig.
        prompts:     list of prompts for this iteration (already replicated).
        global_step: for logging.
        generator:   optional RNG for reproducibility.

    Returns:
        Metrics dict.
    """
    device = next(pipe.unet.parameters()).device

    # ── 1. Encode prompts ────────────────────────────────────────────────
    prompt_embeds = encode_prompt(pipe, prompts).to(device)

    # ── 2. Rollout (no grad) ─────────────────────────────────────────────
    pipe.unet.eval()
    data = rollout(
        pipe,
        prompt_embeds,
        num_inference_steps=cfg.num_inference_steps,
        eta=cfg.ddim_eta,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
        image_size=cfg.image_size,
    )

    # ── 3. Decode → images ───────────────────────────────────────────────
    final_latents = data.latents[-1].to(device)
    images: List[Image.Image] = decode_latents(pipe, final_latents)

    # ── 4. Score ─────────────────────────────────────────────────────────
    rewards = reward_fn(images, prompts)  # (B,) CPU tensor

    # ── 5. Advantages ────────────────────────────────────────────────────
    advantages = compute_advantages(rewards, clip_value=cfg.advantage_clip)

    # ── 6. Inner PPO epochs ──────────────────────────────────────────────
    all_metrics: Dict[str, float] = {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
    }

    for epoch in range(cfg.inner_epochs):
        metrics = ppo_step(
            pipe,
            data,
            advantages,
            data.log_probs,
            optimizer,
            cfg,
            global_step,
        )
        for k, v in metrics.items():
            all_metrics[f"epoch{epoch}/{k}"] = v

    return all_metrics
