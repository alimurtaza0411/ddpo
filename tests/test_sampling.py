"""
Tests for the DDIM sampling + log-probability module.

These are the most critical tests:  the log-prob computation is the #1 source
of silent bugs in DDPO implementations.
"""

from __future__ import annotations

import math

import torch

from ddpo.sampling import ddim_step_with_logprob


# ── Test 1: log-prob consistency ─────────────────────────────────────────────

def test_log_prob_consistency(tiny_pipe, device):
    """
    Rollout generates x_{t-1} with stored log_prob.
    Recomputing log_prob of that same x_{t-1} under the same UNet state
    must match to within 1e-4.
    """
    pipe = tiny_pipe
    pipe.unet.eval()
    pipe.scheduler.set_timesteps(20, device=device)

    B, C, H, W = 2, 4, 16, 16
    cross_dim = 32
    seq_len = 5

    prompt_embeds = torch.randn(B, seq_len, cross_dim, device=device)
    x_t = torch.randn(B, C, H, W, device=device)

    # Pick a mid-range timestep
    t = int(pipe.scheduler.timesteps[5])

    # Rollout mode: get x_{t-1} and log_prob
    with torch.no_grad():
        noise_pred = pipe.unet(x_t, t, encoder_hidden_states=prompt_embeds).sample

    gen = torch.Generator(device=device).manual_seed(123)
    x_prev_rollout, log_prob_rollout = ddim_step_with_logprob(
        pipe.scheduler, noise_pred, t, x_t, eta=1.0, generator=gen,
    )

    # Update mode: recompute log_prob of the SAME x_{t-1}
    with torch.no_grad():
        noise_pred2 = pipe.unet(x_t, t, encoder_hidden_states=prompt_embeds).sample

    _, log_prob_update = ddim_step_with_logprob(
        pipe.scheduler, noise_pred2, t, x_t, eta=1.0, prev_sample=x_prev_rollout,
    )

    # Same UNet state → same noise_pred → same μ, σ → same log_prob
    torch.testing.assert_close(
        log_prob_rollout, log_prob_update, atol=1e-4, rtol=1e-4,
    )


# ── Test 2: log-prob is Gaussian density ─────────────────────────────────────

def test_log_prob_is_gaussian_density():
    """
    Given a fixed μ and σ, sample 10000 z's, compute empirical log-density
    of the samples, compare to analytical formula.  Tolerance 1%.
    """
    D = 4 * 8 * 8  # C * H * W
    mu = torch.zeros(1, 4, 8, 8)
    sigma_sq = torch.tensor(0.05)
    sigma = torch.sqrt(sigma_sq)

    n_samples = 10000
    torch.manual_seed(42)
    samples = mu + sigma * torch.randn(n_samples, 4, 8, 8)

    # Analytical log-prob for each sample
    diff = samples - mu
    log_probs = (
        -0.5 * (diff ** 2).flatten(1).sum(1) / (sigma_sq + 1e-15)
        - 0.5 * D * torch.log(2.0 * math.pi * sigma_sq + 1e-15)
    )

    # Empirical: mean log-prob should equal the entropy of N(0, σ²I)
    # E[log N(x; μ, σ²I)] = -D/2 * (1 + log(2πσ²))
    expected_mean = -0.5 * D * (1.0 + math.log(2.0 * math.pi * sigma_sq.item()))
    empirical_mean = log_probs.mean().item()

    rel_error = abs(empirical_mean - expected_mean) / abs(expected_mean)
    assert rel_error < 0.01, (
        f"Empirical mean log-prob {empirical_mean:.4f} vs expected {expected_mean:.4f}, "
        f"relative error {rel_error:.4f} > 1%"
    )


# ── Test 5: PPO ratio at init ───────────────────────────────────────────────

def test_ppo_ratio_at_init(tiny_pipe, device):
    """
    On the first update epoch before any gradient step, ratio =
    exp(new_log_prob - old_log_prob) must equal 1.0 exactly (same network state).

    If this fails, there's non-determinism in the UNet (dropout, etc.).
    """
    pipe = tiny_pipe
    pipe.unet.eval()
    pipe.scheduler.set_timesteps(10, device=device)

    B, C, H, W = 2, 4, 16, 16
    cross_dim = 32
    seq_len = 5

    prompt_embeds = torch.randn(B, seq_len, cross_dim, device=device)
    x_t = torch.randn(B, C, H, W, device=device)
    t = int(pipe.scheduler.timesteps[3])

    # Rollout: get x_{t-1} and old_log_prob
    with torch.no_grad():
        noise_pred = pipe.unet(x_t, t, encoder_hidden_states=prompt_embeds).sample

    gen = torch.Generator(device=device).manual_seed(99)
    x_prev, old_log_prob = ddim_step_with_logprob(
        pipe.scheduler, noise_pred, t, x_t, eta=1.0, generator=gen,
    )

    # Update: recompute with same UNet (no gradient step happened)
    with torch.no_grad():
        noise_pred2 = pipe.unet(x_t, t, encoder_hidden_states=prompt_embeds).sample

    _, new_log_prob = ddim_step_with_logprob(
        pipe.scheduler, noise_pred2, t, x_t, eta=1.0, prev_sample=x_prev,
    )

    ratio = torch.exp(new_log_prob - old_log_prob)
    torch.testing.assert_close(
        ratio,
        torch.ones_like(ratio),
        atol=1e-5,
        rtol=1e-5,
    )


# ── Test 6: rollout determinism ──────────────────────────────────────────────

def test_rollout_determinism(tiny_pipe, device):
    """Same seed → same latents → same log_probs."""
    pipe = tiny_pipe
    pipe.unet.eval()
    pipe.scheduler.set_timesteps(10, device=device)

    B, C, H, W = 1, 4, 16, 16
    cross_dim = 32
    seq_len = 5

    prompt_embeds = torch.randn(B, seq_len, cross_dim, device=device)

    def run_trajectory(seed):
        pipe.scheduler.set_timesteps(10, device=device)
        gen = torch.Generator(device=device).manual_seed(seed)
        x_t = torch.randn(B, C, H, W, generator=gen, device=device)

        latents = [x_t.clone()]
        log_probs = []

        gen2 = torch.Generator(device=device).manual_seed(seed + 1000)
        for t in pipe.scheduler.timesteps:
            t_int = int(t)
            with torch.no_grad():
                noise_pred = pipe.unet(
                    x_t, t_int, encoder_hidden_states=prompt_embeds
                ).sample
            x_t, lp = ddim_step_with_logprob(
                pipe.scheduler, noise_pred, t_int, x_t, eta=1.0, generator=gen2,
            )
            latents.append(x_t.clone())
            log_probs.append(lp.clone())

        return latents, log_probs

    latents1, lps1 = run_trajectory(42)
    latents2, lps2 = run_trajectory(42)

    for i, (l1, l2) in enumerate(zip(latents1, latents2)):
        torch.testing.assert_close(l1, l2, atol=0, rtol=0)

    for i, (lp1, lp2) in enumerate(zip(lps1, lps2)):
        torch.testing.assert_close(lp1, lp2, atol=0, rtol=0)
