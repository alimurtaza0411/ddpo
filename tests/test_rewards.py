"""Tests for multi-objective reward composition."""

from __future__ import annotations

import torch

from ddpo.rewards import MultiObjectiveReward


class FakeDiversity:
    def per_image_novelty(self, images, prompts):
        return torch.tensor([0.1, 0.3, 0.2, 0.4], dtype=torch.float32)


class FakeFaithfulness:
    def score_text_image(self, images, prompts):
        return torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)


def fake_preference(images, prompts):
    return torch.tensor([10.0, 11.0, 12.0, 13.0], dtype=torch.float32)


def test_multi_objective_reward_logs_components():
    reward = MultiObjectiveReward(
        preference_fn=fake_preference,
        weights={"preference": 1.0, "diversity": 0.5, "faithfulness": 0.25},
        diversity_scorer=FakeDiversity(),
        faithfulness_scorer=FakeFaithfulness(),
        normalize_components=True,
    )

    out = reward([object()] * 4, ["a", "a", "b", "b"])

    assert out.total.shape == (4,)
    assert set(out.components) == {"preference", "diversity", "faithfulness"}
    assert "reward/preference_raw_mean" in out.metrics
    assert "reward/diversity_raw_mean" in out.metrics
    assert "reward/faithfulness_raw_mean" in out.metrics
    assert torch.isfinite(out.total).all()


def test_multi_objective_reward_can_use_raw_scales():
    reward = MultiObjectiveReward(
        preference_fn=fake_preference,
        weights={"preference": 1.0, "diversity": 2.0},
        diversity_scorer=FakeDiversity(),
        normalize_components=False,
    )

    out = reward([object()] * 4, ["a", "a", "b", "b"])
    expected = torch.tensor([10.2, 11.6, 12.4, 13.8])
    torch.testing.assert_close(out.total, expected)
