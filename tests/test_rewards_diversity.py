"""Tests for DiversityReward (Phase 2 task 4).

Diversity is across-batch and per-prompt-group; this is the trickiest
reward to get right. Tests use a deterministic fake LPIPS so the reduction
math (per-row mean of off-diagonal pairwise distances) can be verified
exactly without loading the real ~25 MB LPIPS network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ddpo.rewards import DiversityReward
from ddpo.reward_ensemble import REWARD_REGISTRY


# ── Fake LPIPS implementations ───────────────────────────────────────────────

class _ConstLPIPS:
    """Returns a constant distance for every pair (used to verify reduction).

    Critical: the real LPIPS reports d(x, x) ≈ 0 for identical images. A naïve
    mean over ALL pairs (including the diagonal) would underestimate the true
    diversity. We verify the implementation correctly EXCLUDES the diagonal
    even when our fake reports a non-zero diagonal — that catches the bug.
    """

    def __init__(self, dist: float):
        self.dist = dist

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        N = a.shape[0]
        return torch.full((N, 1, 1, 1), self.dist, dtype=torch.float32)


class _IdentityAwareLPIPS:
    """d(x, x) = 0; otherwise the user-supplied callable on (a, b)."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, a, b):
        N = a.shape[0]
        out = torch.zeros(N, 1, 1, 1, dtype=torch.float32)
        for k in range(N):
            if torch.equal(a[k], b[k]):
                out[k] = 0.0
            else:
                out[k] = self.fn(a[k], b[k])
        return out


def _make(model):
    return DiversityReward(device="cpu", _model=model)


def _img(value: float, shape=(3, 4, 4)) -> np.ndarray:
    """A flat-colored uint8 image."""
    return np.full(shape[1:] + (3,), int(value), dtype=np.uint8)


# ── Reduction correctness ────────────────────────────────────────────────────

def test_constant_distance_with_diagonal_artefact_is_excluded():
    """If LPIPS reports d=0.5 for every pair (including the diagonal), the
    correct mean over off-diagonal is also 0.5. A buggy implementation that
    averages over ALL G*G pairs (including the i==i diagonal) would yield
    0.5 * (G−1)/G. We assert the off-diagonal mean."""
    fake = _ConstLPIPS(0.5)
    drw = _make(fake)
    images = [_img(0), _img(50), _img(100), _img(150)]
    out = drw.score(images, prompts=["p"] * 4)
    assert torch.allclose(out, torch.full((4,), 0.5)), out


def test_singleton_group_returns_zero():
    fake = _ConstLPIPS(0.7)
    drw = _make(fake)
    out = drw.score([_img(0)], ["only-prompt"])
    assert out.shape == (1,)
    assert out.item() == 0.0


def test_each_prompt_group_handled_independently():
    """Two prompts: group A (3 samples), group B (1 sample, singleton)."""
    fake = _ConstLPIPS(0.4)
    drw = _make(fake)
    images = [_img(0), _img(50), _img(100), _img(200)]
    prompts = ["A", "A", "A", "B"]
    out = drw.score(images, prompts)
    expected = torch.tensor([0.4, 0.4, 0.4, 0.0])
    assert torch.allclose(out, expected), out


def test_two_identical_images_same_prompt_diversity_zero():
    """When samples are identical and the (real-style) LPIPS reports 0, all
    rewards in the group should be 0."""
    fake = _IdentityAwareLPIPS(lambda a, b: torch.tensor(0.0))
    drw = _make(fake)
    img = _img(128)
    out = drw.score([img, img, img], ["p"] * 3)
    assert torch.allclose(out, torch.zeros(3)), out


def test_one_outlier_sample_gets_higher_diversity():
    """Sample 0 differs from sample 1 a lot; sample 2 matches sample 1.
    Expected per-row means (off-diag, dist(i,j)):
       i=0: mean(d(0,1), d(0,2)) = mean(1.0, 1.0) = 1.0
       i=1: mean(d(1,0), d(1,2)) = mean(1.0, 0.0) = 0.5
       i=2: mean(d(2,0), d(2,1)) = mean(1.0, 0.0) = 0.5
    """
    img_a = _img(0)
    img_b = _img(255)

    def dist(a, b):
        # Distance is 1.0 if average values differ a lot, else 0.0
        if torch.abs(a.mean() - b.mean()) > 0.1:
            return torch.tensor(1.0)
        return torch.tensor(0.0)

    fake = _IdentityAwareLPIPS(dist)
    drw = _make(fake)
    out = drw.score([img_a, img_b, img_b], ["p"] * 3)
    assert torch.allclose(out, torch.tensor([1.0, 0.5, 0.5])), out


def test_order_within_group_does_not_matter():
    """Reward at index i should depend only on prompt group membership, not order."""
    fake = _ConstLPIPS(0.3)
    drw = _make(fake)
    a = drw.score([_img(0), _img(50), _img(100)], ["p"] * 3)
    b = drw.score([_img(100), _img(0), _img(50)], ["p"] * 3)
    assert torch.allclose(a, b)


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_returns_cpu_float32_tensor_of_correct_shape():
    fake = _ConstLPIPS(0.2)
    drw = _make(fake)
    out = drw.score([_img(0), _img(255), _img(50)], ["p"] * 3)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
    assert out.shape == (3,)


def test_empty_inputs_return_empty_tensor():
    fake = _ConstLPIPS(0.5)
    drw = _make(fake)
    out = drw.score([], [])
    assert out.shape == (0,)


def test_mismatched_lengths_raise():
    fake = _ConstLPIPS(0.5)
    drw = _make(fake)
    with pytest.raises(ValueError, match="must be the same length"):
        drw.score([_img(0)], ["p", "q"])


def test_all_singleton_groups_zero():
    fake = _ConstLPIPS(0.5)
    drw = _make(fake)
    out = drw.score([_img(0), _img(50), _img(100)], ["p1", "p2", "p3"])
    assert torch.allclose(out, torch.zeros(3))


def test_accepts_torch_tensor_input():
    fake = _ConstLPIPS(0.5)
    drw = _make(fake)
    # batch of (3, 4, 4) in [0, 1]
    imgs = torch.rand(3, 3, 4, 4)
    out = drw.score(imgs, ["p", "p", "p"])
    assert out.shape == (3,) and torch.allclose(out, torch.full((3,), 0.5))


def test_diversity_registered():
    assert "diversity" in REWARD_REGISTRY


def test_plays_well_with_ensemble():
    fake = _ConstLPIPS(0.5)
    drw = _make(fake)
    from ddpo.reward_ensemble import RewardEnsemble
    ens = RewardEnsemble([(drw.score, 0.6, "div")])
    combined, per = ens(images=[_img(0), _img(255), _img(50), _img(200)], prompts=["p"] * 4)
    assert torch.allclose(combined, torch.full((4,), 0.3))     # 0.5 * 0.6
    assert torch.allclose(per["div"], torch.full((4,), 0.5))


# ── Hygiene (regression sentinels) ───────────────────────────────────────────

def test_does_not_mix_groups():
    """Rewards must not leak between prompts: same image in two different groups
    should be treated as a singleton in each."""
    fake = _ConstLPIPS(0.7)
    drw = _make(fake)
    out = drw.score([_img(0), _img(0)], ["p1", "p2"])
    assert torch.allclose(out, torch.zeros(2))   # both singletons across groups


@pytest.mark.parametrize("G", [2, 3, 4, 5])
def test_const_distance_recovered_for_any_group_size(G):
    fake = _ConstLPIPS(0.123)
    drw = _make(fake)
    images = [_img(i * 10) for i in range(G)]
    out = drw.score(images, ["p"] * G)
    assert torch.allclose(out, torch.full((G,), 0.123)), out
