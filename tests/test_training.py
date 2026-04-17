"""
Tests for the training module: advantages, LoRA isolation, PPO loss.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model

from ddpo.training import compute_advantages


# ── Test 3: advantage zero mean ──────────────────────────────────────────────

def test_advantage_zero_mean():
    """
    For any batch, advantages after normalisation should have mean ≈ 0
    and std ≈ 1 (before clamping).
    """
    torch.manual_seed(0)
    for _ in range(10):
        rewards = torch.randn(32) * 5 + 3  # random mean & scale
        adv = compute_advantages(rewards, clip_value=100.0)  # large clip → no clamping

        assert abs(adv.mean().item()) < 1e-5, (
            f"Advantage mean {adv.mean().item():.6f} is not ≈ 0"
        )
        assert abs(adv.std().item() - 1.0) < 0.05, (
            f"Advantage std {adv.std().item():.6f} is not ≈ 1"
        )


# ── Test 4: LoRA only trains ────────────────────────────────────────────────

def test_lora_only_trains(tiny_unet, device):
    """
    After one gradient step, frozen UNet base weights are bit-exact unchanged;
    LoRA weights have changed.
    """
    unet = tiny_unet.to(device)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)

    # Cast LoRA params to float32
    for name, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # Snapshot base weights
    base_snapshots = {}
    lora_snapshots = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            lora_snapshots[name] = param.data.clone()
        else:
            base_snapshots[name] = param.data.clone()

    # Forward + backward
    B, C, H, W = 2, 4, 16, 16
    cross_dim = 32
    seq_len = 5
    x = torch.randn(B, C, H, W, device=device)
    t = torch.tensor([500], device=device).expand(B)
    enc = torch.randn(B, seq_len, cross_dim, device=device)

    unet.train()
    out = unet(x, t, encoder_hidden_states=enc).sample
    loss = out.mean()
    loss.backward()

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=1e-3,
    )
    optimizer.step()

    # Check: base weights unchanged
    for name, param in unet.named_parameters():
        if not param.requires_grad:
            if name in base_snapshots:
                assert torch.equal(param.data, base_snapshots[name]), (
                    f"Base weight {name} changed after gradient step!"
                )

    # Check: LoRA weights changed
    any_changed = False
    for name, param in unet.named_parameters():
        if param.requires_grad:
            if name in lora_snapshots:
                if not torch.equal(param.data, lora_snapshots[name]):
                    any_changed = True
                    break

    assert any_changed, "No LoRA weight changed after gradient step!"
