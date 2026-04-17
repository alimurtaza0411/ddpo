"""
PickScore reward model wrapper.

PickScore uses CLIP-ViT-H (laion/CLIP-ViT-H-14-laion2B-s32B-b79K),
NOT the CLIP-ViT-L that Stable Diffusion 1.5 uses.  We load a separate
processor and model from the PickScore checkpoint.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor


class PickScoreReward(nn.Module):
    """Wraps the PickScore v1 model for batch image scoring."""

    def __init__(
        self,
        model_id: str = "yuvalkirstain/PickScore_v1",
        processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(processor_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(device, dtype=dtype)

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def score(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Compute PickScore for each (image, prompt) pair.

        Args:
            images:  list of B PIL images.
            prompts: list of B text prompts (one per image).

        Returns:
            scores: (B,) float32 tensor on CPU.
        """
        image_inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device, dtype=self.dtype)

        text_inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        # PickScore = logit_scale * cosine_similarity
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (image_embs * text_embs).sum(dim=-1)

        return scores.float().cpu()
