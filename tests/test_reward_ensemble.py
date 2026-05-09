"""Tests for the reward-ensemble plumbing (Phase 2 task 1)."""

from __future__ import annotations

import pytest
import torch

from ddpo.reward_ensemble import (
    REWARD_REGISTRY,
    RewardEnsemble,
    build_ensemble,
    coerce_reward,
    register_reward,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _const_reward(value: float):
    """Return a fake reward callable that ignores inputs and yields a constant."""
    def fn(images, prompts):
        return torch.full((len(prompts),), value, dtype=torch.float32)
    return fn


def _proportional_reward(scale: float):
    def fn(images, prompts):
        return scale * torch.arange(len(prompts), dtype=torch.float32)
    return fn


# ── single-component / backwards compat ──────────────────────────────────────

def test_single_component_combined_equals_weighted_component():
    fn = _const_reward(2.0)
    ens = RewardEnsemble([(fn, 1.5, "x")])
    combined, per = ens(images=[None] * 4, prompts=["p"] * 4)
    assert combined.shape == (4,)
    assert torch.allclose(combined, torch.full((4,), 3.0))   # 2.0 * 1.5
    assert "x" in per and torch.allclose(per["x"], torch.full((4,), 2.0))  # raw, unweighted


def test_coerce_callable_wraps_to_single_component():
    fn = _const_reward(7.0)
    ens = coerce_reward(fn)
    assert isinstance(ens, RewardEnsemble)
    assert len(ens) == 1
    assert ens.names == ["reward"]
    assert ens.weights == [1.0]
    combined, per = ens([None] * 3, ["p"] * 3)
    assert torch.allclose(combined, per["reward"])


def test_coerce_passes_through_existing_ensemble():
    ens1 = RewardEnsemble([(_const_reward(1.0), 1.0, "a")])
    ens2 = coerce_reward(ens1)
    assert ens2 is ens1


def test_coerce_rejects_non_callable():
    with pytest.raises(TypeError):
        coerce_reward(42)  # type: ignore[arg-type]


# ── multi-component math ─────────────────────────────────────────────────────

def test_combined_is_weighted_sum_of_components():
    a = _const_reward(1.0)
    b = _proportional_reward(2.0)  # values 0, 2, 4, 6 for B=4
    ens = RewardEnsemble([(a, 0.5, "a"), (b, 0.25, "b")])

    combined, per = ens([None] * 4, ["p"] * 4)
    expected = 0.5 * torch.full((4,), 1.0) + 0.25 * torch.tensor([0.0, 2.0, 4.0, 6.0])
    assert torch.allclose(combined, expected)
    # Per-component is RAW (unweighted) — that's the contract
    assert torch.allclose(per["a"], torch.full((4,), 1.0))
    assert torch.allclose(per["b"], torch.tensor([0.0, 2.0, 4.0, 6.0]))


def test_zero_weight_zeros_contribution_but_keeps_logging():
    a = _const_reward(5.0)
    b = _const_reward(7.0)
    ens = RewardEnsemble([(a, 1.0, "keep"), (b, 0.0, "muted")])
    combined, per = ens([None] * 2, ["p"] * 2)
    assert torch.allclose(combined, torch.full((2,), 5.0))   # only "keep" contributes
    assert torch.allclose(per["muted"], torch.full((2,), 7.0))  # but raw signal still logged


# ── input validation ─────────────────────────────────────────────────────────

def test_empty_ensemble_rejected():
    with pytest.raises(ValueError, match="at least one"):
        RewardEnsemble([])


def test_duplicate_names_rejected():
    a = _const_reward(1.0)
    with pytest.raises(ValueError, match="Duplicate"):
        RewardEnsemble([(a, 1.0, "x"), (a, 1.0, "x")])


def test_non_tensor_returns_are_coerced():
    """Reward fns may return numpy arrays, lists, etc.; ensemble normalizes."""
    def list_fn(images, prompts):
        return [1.0, 2.0, 3.0]
    ens = RewardEnsemble([(list_fn, 1.0, "x")])
    combined, per = ens([None] * 3, ["p"] * 3)
    assert torch.allclose(combined, torch.tensor([1.0, 2.0, 3.0]))


# ── registry / config-driven construction ────────────────────────────────────

def test_register_and_build_from_config():
    # Register a fresh fake type for this test (don't pollute the global registry)
    name = "_fake_in_test"
    if name in REWARD_REGISTRY:
        del REWARD_REGISTRY[name]

    @register_reward(name)
    def _factory(cfg):
        return _const_reward(cfg.get("value", 9.0))

    try:
        ens = build_ensemble([{"type": name, "weight": 0.5}])
        assert ens.names == [name]
        assert ens.weights == [0.5]
        combined, per = ens([None] * 2, ["p"] * 2)
        assert torch.allclose(combined, torch.full((2,), 4.5))   # 9.0 * 0.5
        assert torch.allclose(per[name], torch.full((2,), 9.0))  # raw

        ens2 = build_ensemble(
            [{"type": name, "weight": 1.0, "name": "alias", "config": {"value": 2.0}}]
        )
        assert ens2.names == ["alias"]
        combined2, per2 = ens2([None] * 1, ["p"])
        assert torch.allclose(combined2, torch.full((1,), 2.0))
    finally:
        REWARD_REGISTRY.pop(name, None)


def test_build_ensemble_unknown_type_raises():
    with pytest.raises(KeyError, match="not registered"):
        build_ensemble([{"type": "definitely_not_a_real_reward", "weight": 1.0}])


def test_register_duplicate_rejected():
    name = "_dup_in_test"
    REWARD_REGISTRY.pop(name, None)
    @register_reward(name)
    def _factory(cfg):
        return _const_reward(0.0)
    try:
        with pytest.raises(ValueError, match="already registered"):
            @register_reward(name)
            def _factory2(cfg):
                return _const_reward(1.0)
    finally:
        REWARD_REGISTRY.pop(name, None)


def test_pickscore_is_registered():
    # Lazy-import to ensure rewards.py runs its registration block.
    import ddpo.rewards  # noqa: F401
    assert "pickscore" in REWARD_REGISTRY


# ── shape preservation ───────────────────────────────────────────────────────

@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_combined_shape_matches_batch(B):
    ens = RewardEnsemble([(_const_reward(1.0), 1.0, "x"), (_const_reward(2.0), 0.5, "y")])
    combined, per = ens([None] * B, ["p"] * B)
    assert combined.shape == (B,)
    assert per["x"].shape == (B,)
    assert per["y"].shape == (B,)


# ── reward_summary_metrics: weighted_mean = weight × raw_mean ────────────────

def test_reward_summary_metrics_weighted_equals_weight_times_raw():
    """The locked design contract: weighted_mean is computed from raw_mean,
    not re-derived from the data. Verifying this as a unit test catches any
    future re-introduction of two divergent code paths.
    """
    from ddpo.training import reward_summary_metrics

    per_component = {
        "pickscore": torch.tensor([18.0, 20.0, 22.0]),   # raw mean = 20.0
        "diversity": torch.tensor([0.30, 0.40, 0.50]),    # raw mean = 0.40
        "clipscore": torch.tensor([0.25, 0.30, 0.35]),    # raw mean = 0.30
    }
    weights = {"pickscore": 1.0, "diversity": 20.0, "clipscore": 20.0}
    combined = sum(w * per_component[n] for n, w in weights.items())

    m = reward_summary_metrics(combined, per_component, weights)

    # Combined drives the advantage
    assert m["reward_mean"] == pytest.approx(combined.mean().item())
    assert m["reward_std"] == pytest.approx(combined.std().item())

    # Per-component raw stats
    assert m["reward/pickscore_raw_mean"] == pytest.approx(20.0)
    assert m["reward/diversity_raw_mean"] == pytest.approx(0.40)
    assert m["reward/clipscore_raw_mean"] == pytest.approx(0.30)

    # The headline invariant: weighted = weight × raw
    for name in ("pickscore", "diversity", "clipscore"):
        raw = m[f"reward/{name}_raw_mean"]
        weighted = m[f"reward/{name}_weighted_mean"]
        assert weighted == pytest.approx(weights[name] * raw), (
            f"{name}: weighted {weighted} != weight {weights[name]} × raw {raw}"
        )


def test_reward_summary_metrics_zero_weight_zeros_weighted_mean():
    """A component with weight 0 has weighted_mean == 0, raw_mean unaffected."""
    from ddpo.training import reward_summary_metrics
    per_component = {"x": torch.tensor([5.0, 7.0])}
    metrics = reward_summary_metrics(torch.zeros(2), per_component, {"x": 0.0})
    assert metrics["reward/x_raw_mean"] == pytest.approx(6.0)
    assert metrics["reward/x_weighted_mean"] == 0.0
