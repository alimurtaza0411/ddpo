"""
Diversity metrics for intra-prompt image diversity.

Two complementary metrics:
  1. LPIPS — mean pairwise perceptual distance (pixel-level diversity)
  2. CLIP embedding variance — spread of CLIP image embeddings (semantic diversity)

For a set of N images generated from the same prompt:
    LPIPS diversity  = (2 / (N*(N-1))) Σ_{i<j} LPIPS(img_i, img_j)
    CLIP variance    = mean of per-dimension variances of CLIP embeddings
"""

from __future__ import annotations

import logging
from typing import List

import lpips
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)


class LPIPSDiversity:
    """Compute mean pairwise LPIPS distance for a batch of images."""

    def __init__(self, net: str = "alex", device: str = "cuda"):
        self.device = device
        self.model = lpips.LPIPS(net=net).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def compute(self, images: List[Image.Image]) -> float:
        """
        Mean pairwise LPIPS for a set of images.

        Args:
            images: list of N PIL images (same prompt, different seeds).

        Returns:
            Mean pairwise LPIPS distance (scalar).
        """
        n = len(images)
        if n < 2:
            return 0.0

        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.model(
                    tensors[i : i + 1], tensors[j : j + 1]
                ).item()
                total += dist
                count += 1

        return total / count


class CLIPDiversity:
    """
    Compute CLIP embedding variance for a batch of images.

    For N images from the same prompt, embeds each image into CLIP space
    and computes the mean per-dimension variance of the L2-normalised
    embeddings.  Higher variance → more semantically diverse outputs.

    Uses CLIP-ViT-L/14 (same architecture as SD 1.5's image understanding).
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        logger.info("Loading CLIP model for diversity: %s", model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute(self, images: List[Image.Image]) -> float:
        """
        CLIP embedding variance for a set of images.

        Args:
            images: list of N PIL images (same prompt, different seeds).

        Returns:
            Mean per-dimension variance of L2-normalised CLIP embeddings (scalar).
        """
        n = len(images)
        if n < 2:
            return 0.0

        # Process in batches to avoid OOM with many images
        batch_size = 16
        all_embs = []
        for start in range(0, n, batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(
                images=batch, return_tensors="pt",
            ).to(self.device)
            embs = self.model.get_image_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.float().cpu())

        embeddings = torch.cat(all_embs, dim=0)  # (N, D)

        # Mean of per-dimension variances
        variance = embeddings.var(dim=0).mean().item()
        return variance
