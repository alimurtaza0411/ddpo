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
from typing import Dict, List

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

    @torch.no_grad()
    def per_image_novelty(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Mean LPIPS distance from each image to same-prompt siblings.

        Returns a CPU tensor with one reward per image.  This is useful as a
        non-differentiable RL reward: images that are perceptually different
        from other samples for the same prompt receive a larger value.
        """
        n = len(images)
        if n < 2:
            return torch.zeros(n, dtype=torch.float32)

        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        rewards = torch.zeros(n, dtype=torch.float32)

        prompt_to_indices: Dict[str, List[int]] = {}
        for idx, prompt in enumerate(prompts):
            prompt_to_indices.setdefault(prompt, []).append(idx)

        for indices in prompt_to_indices.values():
            if len(indices) < 2:
                continue

            totals = {idx: 0.0 for idx in indices}
            counts = {idx: 0 for idx in indices}
            for pos, i in enumerate(indices):
                for j in indices[pos + 1:]:
                    dist = self.model(
                        tensors[i : i + 1], tensors[j : j + 1]
                    ).item()
                    totals[i] += dist
                    totals[j] += dist
                    counts[i] += 1
                    counts[j] += 1

            for idx in indices:
                if counts[idx] > 0:
                    rewards[idx] = totals[idx] / counts[idx]

        return rewards


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
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        L2-normalised CLIP image embeddings on CPU.

        Args:
            images: list of N PIL images.

        Returns:
            Tensor with shape (N, D).
        """
        batch_size = 16
        all_embs = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(
                images=batch, return_tensors="pt",
            ).to(self.device)
            embs = self.model.get_image_features(**inputs)
            # Handle BaseModelOutputWithPooling (extract raw tensor)
            embs = getattr(embs, "image_embeds",
                           getattr(embs, "pooler_output", embs))
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.float().cpu())

        return torch.cat(all_embs, dim=0)

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

        embeddings = self.embed_images(images)

        # Mean of per-dimension variances
        variance = embeddings.var(dim=0).mean().item()
        return variance

    @torch.no_grad()
    def per_image_novelty(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Mean cosine distance from each image to same-prompt siblings.

        Because image embeddings are L2-normalised, cosine distance is
        ``1 - dot(e_i, e_j)``.  Higher values reward semantically different
        outputs for the same prompt.
        """
        n = len(images)
        if n < 2:
            return torch.zeros(n, dtype=torch.float32)

        embeddings = self.embed_images(images)
        rewards = torch.zeros(n, dtype=torch.float32)

        prompt_to_indices: Dict[str, List[int]] = {}
        for idx, prompt in enumerate(prompts):
            prompt_to_indices.setdefault(prompt, []).append(idx)

        for indices in prompt_to_indices.values():
            if len(indices) < 2:
                continue

            group = embeddings[indices]
            sim = group @ group.T
            dist = 1.0 - sim
            dist.fill_diagonal_(0.0)
            rewards[indices] = dist.sum(dim=1) / (len(indices) - 1)

        return rewards

    @torch.no_grad()
    def score_text_image(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> torch.Tensor:
        """
        CLIP image-text cosine similarity for each (image, prompt) pair.

        This is a lightweight prompt-faithfulness proxy for multi-objective
        experiments.  It is not a replacement for GenEval, but it gives the
        policy a separate prompt-alignment term without loading another model.
        """
        image_embs = self.embed_images(images).to(self.device)
        text_inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = getattr(text_embs, "text_embeds",
                           getattr(text_embs, "pooler_output", text_embs))
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        scores = (image_embs * text_embs).sum(dim=-1)
        return scores.float().cpu()
