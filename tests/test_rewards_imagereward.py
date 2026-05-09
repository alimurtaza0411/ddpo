"""Tests for the ImageRewardModel wrapper (Phase 2 task 2).

We don't load the real ~700 MB ImageReward checkpoint in unit tests — we inject
a fake scorer via the `_model=` test seam.
"""

from __future__ import annotations

import pytest
import torch

from ddpo.rewards import ImageRewardModel
from ddpo.reward_ensemble import REWARD_REGISTRY


class _FakeRM:
    """Stand-in for the upstream `ImageReward` model.

    `.score(prompt, image)` returns a deterministic float so we can assert
    exactly what the wrapper produces.
    """

    def __init__(self, score_fn=None):
        self.calls: list[tuple[str, object]] = []
        self.score_fn = score_fn or (lambda p, i: float(len(p)))

    def score(self, prompt, image):
        self.calls.append((prompt, image))
        return self.score_fn(prompt, image)


def test_wrapper_loops_over_batch_in_order():
    fake = _FakeRM(score_fn=lambda p, i: float(len(p)))
    irm = ImageRewardModel(_model=fake)
    out = irm.score(images=["img0", "img1", "img2"], prompts=["a", "ab", "abc"])
    assert out.dtype == torch.float32
    assert out.shape == (3,)
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]))
    assert fake.calls == [("a", "img0"), ("ab", "img1"), ("abc", "img2")]


def test_wrapper_returns_cpu_tensor():
    fake = _FakeRM(score_fn=lambda p, i: 0.5)
    irm = ImageRewardModel(_model=fake)
    out = irm.score(["i"] * 4, ["p"] * 4)
    assert out.device.type == "cpu"


def test_mismatched_lengths_raise():
    fake = _FakeRM()
    irm = ImageRewardModel(_model=fake)
    with pytest.raises(ValueError, match="must be the same length"):
        irm.score(images=["a", "b"], prompts=["x"])


def test_empty_inputs_return_empty_tensor():
    fake = _FakeRM()
    irm = ImageRewardModel(_model=fake)
    out = irm.score(images=[], prompts=[])
    assert out.shape == (0,)


def test_real_model_path_raises_if_library_missing(monkeypatch):
    """If `import ImageReward` fails, we surface a clear error."""
    import builtins as _bi
    real_import = _bi.__import__

    def _block(name, *args, **kwargs):
        if name == "ImageReward":
            raise ImportError("simulated: ImageReward not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_bi, "__import__", _block)
    with pytest.raises(ImportError, match="image-reward"):
        ImageRewardModel(model_id="fake")  # _model not provided -> tries real load


def test_imagereward_registered():
    assert "imagereward" in REWARD_REGISTRY


def test_score_results_can_be_consumed_by_ensemble():
    """End-to-end: ImageReward fake plugs into RewardEnsemble cleanly."""
    from ddpo.reward_ensemble import RewardEnsemble

    fake_a = _FakeRM(score_fn=lambda p, i: 1.0)
    fake_b = _FakeRM(score_fn=lambda p, i: 3.0)
    ir_a = ImageRewardModel(_model=fake_a)
    ir_b = ImageRewardModel(_model=fake_b)

    ens = RewardEnsemble([(ir_a.score, 0.5, "a"), (ir_b.score, 0.25, "b")])
    combined, per = ens(["img"] * 4, ["p"] * 4)
    assert torch.allclose(combined, torch.full((4,), 0.5 * 1.0 + 0.25 * 3.0))
    assert torch.allclose(per["a"], torch.full((4,), 1.0))
    assert torch.allclose(per["b"], torch.full((4,), 3.0))
