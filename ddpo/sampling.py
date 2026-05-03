"""
DDIM sampling with per-step log-probability computation.

Implements two modes:
  1. Rollout mode  — sample x_{t-1} fresh, return (x_{t-1}, log_prob)
  2. Update mode   — given stored x_{t-1}, compute log_prob under current policy

Math (DDIM paper, Song et al. 2021, eq. 12):
  x̂_0   = (x_t − √(1−ᾱ_t) · ε_θ) / √(ᾱ_t)
  σ_t²  = η² · (1−ᾱ_{t-1})/(1−ᾱ_t) · (1 − ᾱ_t/ᾱ_{t-1})
  μ     = √(ᾱ_{t-1}) · x̂_0  +  √(1−ᾱ_{t-1}−σ_t²) · ε_θ
  x_{t-1} = μ + σ_t · z,   z ~ N(0, I)
  log π = −‖x_{t-1} − μ‖² / (2 σ_t²)  − D/2 · log(2πσ_t²)
          (summed over spatial dims)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline


@dataclass
class RolloutData:
    """Stores one complete denoising trajectory for a batch."""

    latents: List[torch.Tensor]  # T+1 tensors: x_T, x_{T-1}, …, x_0
    log_probs: List[torch.Tensor]  # T tensors: log π(x_{t-1}|x_t) per step
    prompt_embeds: torch.Tensor  # (B, seq_len, dim)
    timesteps: List[int]  # scheduler timesteps used


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_alpha_bars(
    scheduler: DDIMScheduler,
    timestep: int,
    prev_timestep: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ᾱ_t and ᾱ_{t-1} as scalars on the scheduler device."""
    alpha_bar_t = scheduler.alphas_cumprod[timestep]
    if prev_timestep >= 0:
        alpha_bar_prev = scheduler.alphas_cumprod[prev_timestep]
    else:
        alpha_bar_prev = scheduler.final_alpha_cumprod
    return alpha_bar_t, alpha_bar_prev


def ddim_step_with_logprob(
    scheduler: DDIMScheduler,
    noise_pred: torch.Tensor,
    timestep: int,
    x_t: torch.Tensor,
    eta: float = 1.0,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One DDIM transition with log-probability.

    Args:
        scheduler:   DDIMScheduler with .alphas_cumprod set.
        noise_pred:  ε_θ(x_t, t)  — (B, C, H, W).
        timestep:    current discrete timestep t.
        x_t:         current latent  — (B, C, H, W).
        eta:         DDIM stochasticity.  η=1 → same variance as DDPM.
        prev_sample: if provided, compute log-prob of this sample instead of
                     sampling a new one  (update mode).
        generator:   optional torch.Generator for reproducibility.

    Returns:
        (x_{t-1}, log_prob)  where log_prob has shape (B,).
    """
    # ── 1. Identify previous timestep ────────────────────────────────────
    prev_timestep = (
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    )

    # ── 2. ᾱ_t and ᾱ_{t-1} ──────────────────────────────────────────────
    alpha_bar_t, alpha_bar_prev = _get_alpha_bars(scheduler, timestep, prev_timestep)

    # move to same device/dtype as x_t
    alpha_bar_t = alpha_bar_t.to(x_t.device, dtype=x_t.dtype)
    alpha_bar_prev = alpha_bar_prev.to(x_t.device, dtype=x_t.dtype)

    # ── 3. Predict x_0 ──────────────────────────────────────────────────
    x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(
        alpha_bar_t
    )

    # ── 4. Variance σ_t² ─────────────────────────────────────────────────
    # σ_t² = η² · (1−ᾱ_{t-1}) / (1−ᾱ_t) · (1 − ᾱ_t / ᾱ_{t-1})
    variance = (
        (eta ** 2)
        * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar_t)
        * (1.0 - alpha_bar_t / alpha_bar_prev)
    )
    std_dev = torch.sqrt(variance)

    # ── 5. Direction coefficient (clamp to avoid negative under sqrt) ────
    dir_coeff_sq = 1.0 - alpha_bar_prev - variance
    dir_coeff = torch.sqrt(torch.clamp(dir_coeff_sq, min=0.0))

    # ── 6. Mean μ ────────────────────────────────────────────────────────
    mean = torch.sqrt(alpha_bar_prev) * x0_pred + dir_coeff * noise_pred

    # ── 7. Sample or evaluate ────────────────────────────────────────────
    if prev_sample is None:
        noise = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
        prev_sample_out = mean + std_dev * noise
    else:
        prev_sample_out = prev_sample

    # ── 8. Log-probability ───────────────────────────────────────────────
    #  log N(x; μ, σ²I) = −‖x − μ‖²/(2σ²) − D/2 · log(2πσ²)
    #  We sum over all spatial+channel dims → one scalar per batch element.
    D = prev_sample_out.shape[1:].numel()  # C * H * W
    diff = prev_sample_out - mean
    log_prob = (
        -0.5 * (diff ** 2).flatten(1).sum(1) / (variance + 1e-15)
        - 0.5 * D * torch.log(2.0 * torch.pi * variance + 1e-15)
    )

    return prev_sample_out, log_prob


# ── full trajectory ──────────────────────────────────────────────────────────

@torch.no_grad()
def rollout(
    pipe: StableDiffusionPipeline,
    prompt_embeds: torch.Tensor,
    num_inference_steps: int = 50,
    eta: float = 1.0,
    guidance_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
    image_size: int = 512,
) -> RolloutData:
    """
    Run a full denoising trajectory and store all intermediates.

    Args:
        pipe:               StableDiffusionPipeline with scheduler, unet, vae.
        prompt_embeds:      (B, seq_len, dim) — pre-encoded text.
        num_inference_steps: DDIM steps.
        eta:                DDIM stochasticity.
        guidance_scale:     classifier-free guidance weight (1.0 = no CFG).
        generator:          torch.Generator for reproducibility.
        image_size:         spatial resolution for the latent (latent = image_size/8).

    Returns:
        RolloutData with stored latents, log_probs, prompt_embeds, timesteps.
    """
    device = prompt_embeds.device
    dtype = prompt_embeds.dtype
    batch_size = prompt_embeds.shape[0]
    latent_h = latent_w = image_size // 8

    scheduler: DDIMScheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Initial noise x_T
    latent = torch.randn(
        (batch_size, pipe.unet.config.in_channels, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    all_latents = [latent.detach().cpu()]
    all_log_probs = []
    timestep_list = scheduler.timesteps.tolist()

    pipe.unet.eval()

    for t in scheduler.timesteps:
        t_int = int(t)
        latent_input = latent

        # CFG: if guidance_scale > 1, do unconditional + conditional
        if guidance_scale > 1.0:
            latent_input = torch.cat([latent] * 2)
            noise_pred = pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=torch.cat(
                    [torch.zeros_like(prompt_embeds), prompt_embeds]
                ),
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred = pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

        latent, log_prob = ddim_step_with_logprob(
            scheduler,
            noise_pred,
            t_int,
            latent,
            eta=eta,
            generator=generator,
        )

        all_latents.append(latent.detach().cpu())
        all_log_probs.append(log_prob.detach().cpu())

    return RolloutData(
        latents=all_latents,
        log_probs=all_log_probs,
        prompt_embeds=prompt_embeds.detach().cpu(),
        timesteps=timestep_list,
    )


def compute_log_prob_for_step(
    pipe: StableDiffusionPipeline,
    x_t: torch.Tensor,
    x_prev: torch.Tensor,
    timestep: int,
    prompt_embeds: torch.Tensor,
    eta: float = 1.0,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Re-evaluate log π(x_{t-1} | x_t) under the CURRENT policy (update mode).

    Runs UNet forward WITH gradients so that LoRA params get gradients.

    Returns:
        log_prob: (B,) — differentiable w.r.t. UNet params.
    """
    device = x_t.device

    if guidance_scale > 1.0:
        latent_input = torch.cat([x_t] * 2)
        noise_pred = pipe.unet(
            latent_input,
            timestep,
            encoder_hidden_states=torch.cat(
                [torch.zeros_like(prompt_embeds), prompt_embeds]
            ),
        ).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        noise_pred = pipe.unet(
            x_t,
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample

    _, log_prob = ddim_step_with_logprob(
        pipe.scheduler,
        noise_pred,
        timestep,
        x_t,
        eta=eta,
        prev_sample=x_prev,
    )

    return log_prob


def decode_latents(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> list:
    """Decode latents → PIL images. Uses fp32 for VAE stability."""
    from PIL import Image
    import numpy as np

    latents = latents.to(pipe.vae.device, dtype=torch.float32)
    latents = latents / pipe.vae.config.scaling_factor

    with torch.no_grad():
        imgs = pipe.vae.decode(latents).sample

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype(np.uint8)
    return [Image.fromarray(img) for img in imgs]


def encode_prompt(pipe: StableDiffusionPipeline, prompts: list[str]) -> torch.Tensor:
    """Encode a list of prompts using SD's text encoder. Returns (B, 77, 768)."""
    tok = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = pipe.text_encoder(tok.input_ids.to(pipe.text_encoder.device))
        # Handle both raw tensors and BaseModelOutput objects
        if hasattr(outputs, "last_hidden_state"):
            embeds = outputs.last_hidden_state
        elif isinstance(outputs, (list, tuple)):
            embeds = outputs[0]
        else:
            embeds = outputs
    return embeds
