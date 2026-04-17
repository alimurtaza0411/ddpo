"""
LPIPS-based intra-prompt diversity metric.

For a set of N images generated from the same prompt, diversity is the
mean pairwise LPIPS distance:

    diversity = (2 / (N*(N-1))) Σ_{i<j} LPIPS(img_i, img_j)
"""

from __future__ import annotations

from typing import List

import lpips
import torch
import torchvision.transforms as T
from PIL import Image


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
