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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


@dataclass
class RewardOutput:
    """Composite reward values plus component-level logging data."""

    total: torch.Tensor
    components: Dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


def _standardize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batch-standardise a reward component, with a safe zero-variance path."""
    values = values.float().cpu()
    if values.numel() < 2:
        return torch.zeros_like(values)
    std = values.std()
    if std.item() < eps:
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + eps)


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


class MultiObjectiveReward:
    """
    Compose preference, prompt-faithfulness, and diversity rewards.

    The returned ``total`` tensor is still one scalar per sampled image, so the
    PPO update loop can remain unchanged.  Component values are logged
    separately to support Pareto-style analysis across reward dimensions.
    """

    def __init__(
        self,
        preference_fn: Callable[[List[Image.Image], List[str]], torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
        diversity_scorer: Optional[object] = None,
        faithfulness_scorer: Optional[object] = None,
        image_reward_fn: Optional[Callable[[List[Image.Image], List[str]], torch.Tensor]] = None,
        normalize_components: bool = True,
    ):
        self.preference_fn = preference_fn
        self.weights = {
            "preference": 1.0,
            "faithfulness": 0.0,
            "diversity": 0.0,
            "image_reward": 0.0,
            **(weights or {}),
        }
        self.diversity_scorer = diversity_scorer
        self.faithfulness_scorer = faithfulness_scorer
        self.image_reward_fn = image_reward_fn
        self.normalize_components = normalize_components

    def _component_for_total(self, values: torch.Tensor) -> torch.Tensor:
        if self.normalize_components:
            return _standardize(values)
        return values.float().cpu()

    def __call__(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> RewardOutput:
        components: Dict[str, torch.Tensor] = {}

        preference = self.preference_fn(images, prompts).float().cpu()
        components["preference"] = preference

        if self.weights["faithfulness"] != 0.0:
            if self.faithfulness_scorer is None or not hasattr(
                self.faithfulness_scorer, "score_text_image"
            ):
                raise ValueError(
                    "reward.weights.faithfulness is non-zero, but no scorer "
                    "with score_text_image(images, prompts) was provided."
                )
            components["faithfulness"] = self.faithfulness_scorer.score_text_image(
                images, prompts,
            ).float().cpu()

        if self.weights["diversity"] != 0.0:
            if self.diversity_scorer is None or not hasattr(
                self.diversity_scorer, "per_image_novelty"
            ):
                raise ValueError(
                    "reward.weights.diversity is non-zero, but no scorer "
                    "with per_image_novelty(images, prompts) was provided."
                )
            components["diversity"] = self.diversity_scorer.per_image_novelty(
                images, prompts,
            ).float().cpu()

        if self.weights["image_reward"] != 0.0:
            if self.image_reward_fn is None:
                raise ValueError(
                    "reward.weights.image_reward is non-zero, but ImageReward "
                    "is unavailable."
                )
            components["image_reward"] = self.image_reward_fn(
                images, prompts,
            ).float().cpu()

        total = torch.zeros_like(preference)
        metrics: Dict[str, float] = {}
        for name, values in components.items():
            raw = values.float().cpu()
            weight = float(self.weights.get(name, 0.0))
            total = total + weight * self._component_for_total(raw)
            metrics[f"reward/{name}_raw_mean"] = raw.mean().item()
            metrics[f"reward/{name}_raw_std"] = raw.std().item()
            metrics[f"reward/{name}_weight"] = weight

        metrics["reward/total_mean"] = total.mean().item()
        metrics["reward/total_std"] = total.std().item()

        return RewardOutput(total=total, components=components, metrics=metrics)
