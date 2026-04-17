"""Shared fixtures for DDPO tests."""

from __future__ import annotations

import torch
import pytest


@pytest.fixture
def device():
    """Use CUDA if available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tiny_unet():
    """
    Create a minimal UNet2DConditionModel for fast testing.
    Uses tiny dimensions to keep tests fast on CPU.
    """
    from diffusers import UNet2DConditionModel

    model = UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        norm_num_groups=4,
        attention_head_dim=8,
    )
    model.eval()
    return model


@pytest.fixture
def tiny_pipe(tiny_unet, device):
    """
    Create a minimal pipeline-like object with a tiny UNet and DDIM scheduler.
    Does NOT load SD 1.5 — too slow for unit tests.
    """
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        set_alpha_to_one=False,
    )

    class MinimalPipe:
        def __init__(self, unet, scheduler, device):
            self.unet = unet.to(device)
            self.scheduler = scheduler
            self.device = device

    pipe = MinimalPipe(tiny_unet, scheduler, device)
    return pipe
