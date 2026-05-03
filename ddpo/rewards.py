"""
Reward model wrappers for DDPO evaluation and training.

Two reward models:
  1. PickScore — CLIP-ViT-H based human preference model (primary training reward)
  2. ImageReward — BLIP-based human preference model (additional eval metric)

PickScore uses CLIP-ViT-H (laion/CLIP-ViT-H-14-laion2B-s32B-b79K),
NOT the CLIP-ViT-L that Stable Diffusion 1.5 uses.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor

# --- Compatibility Patch for ImageReward ---
try:
    import transformers.modeling_utils
    if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
        import transformers.pytorch_utils
        transformers.modeling_utils.apply_chunking_to_forward = transformers.pytorch_utils.apply_chunking_to_forward
except (ImportError, AttributeError):
    pass
# -------------------------------------------

logger = logging.getLogger(__name__)


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
        # Some transformers versions return BaseModelOutputWithPooling
        # instead of a raw tensor — extract the tensor defensively.
        image_embs = getattr(image_embs, "image_embeds",
                             getattr(image_embs, "pooler_output", image_embs))
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = getattr(text_embs, "text_embeds",
                            getattr(text_embs, "pooler_output", text_embs))
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        # PickScore = logit_scale * cosine_similarity
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (image_embs * text_embs).sum(dim=-1)

        return scores.float().cpu()


class ImageRewardScore(nn.Module):
    """
    Wraps the ImageReward v1.0 model (THUDM) for batch image scoring.

    ImageReward uses a BLIP-based architecture and is trained on human
    preference data.  Provides a complementary signal to PickScore.

    Used for evaluation only (not as training reward in the baseline).
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        import ImageReward as RM

        self.device = device
        logger.info("Loading ImageReward model...")
        self.model = RM.load("ImageReward-v1.0", device=device)

    @torch.no_grad()
    def score(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Compute ImageReward for each (image, prompt) pair.

        Args:
            images:  list of B PIL images.
            prompts: list of B text prompts (one per image).

        Returns:
            scores: (B,) float32 tensor on CPU.
        """
        scores = []
        for img, prompt in zip(images, prompts):
            s = self.model.score(prompt, img)
            scores.append(s)
        return torch.tensor(scores, dtype=torch.float32)
