"""Tests for the train / held-out prompt split (2026-05-03 refactor).

These are the experimental-design invariants — if any of these fail, the
diversity / faithfulness claims in the paper are no longer well-defined.
"""

from __future__ import annotations

import pytest

from ddpo.prompts import (
    ALL_EVAL_PROMPTS,
    ALL_HELD_OUT_PROMPTS,
    ALL_TRAIN_PROMPTS,
    EVAL_PROMPTS,
    HELD_OUT_PROMPTS,
    TRAIN_PROMPTS,
    get_eval_prompts,
    sample_train_prompts,
)


CATEGORIES = ("constrained", "common_subject", "open_ended")


def test_train_and_held_out_are_string_disjoint():
    overlap = set(ALL_TRAIN_PROMPTS) & set(ALL_HELD_OUT_PROMPTS)
    assert not overlap, f"prompts appear in both train and held-out: {sorted(overlap)}"


def test_train_and_held_out_total_to_56():
    assert len(ALL_TRAIN_PROMPTS) + len(ALL_HELD_OUT_PROMPTS) == 56


def test_train_per_category_counts():
    assert len(TRAIN_PROMPTS["constrained"]) == 12
    assert len(TRAIN_PROMPTS["common_subject"]) == 15
    assert len(TRAIN_PROMPTS["open_ended"]) == 15


def test_held_out_per_category_counts():
    assert len(HELD_OUT_PROMPTS["constrained"]) == 4
    assert len(HELD_OUT_PROMPTS["common_subject"]) == 5
    assert len(HELD_OUT_PROMPTS["open_ended"]) == 5


def test_75_25_split_ratio_per_category():
    """Held-out should be ~25% of each category's pool, ±a few percent."""
    for cat in ("constrained", "common_subject", "open_ended"):
        train_n = len(TRAIN_PROMPTS[cat])
        held_n = len(HELD_OUT_PROMPTS[cat])
        ratio = held_n / (train_n + held_n)
        assert 0.20 <= ratio <= 0.30, (
            f"{cat}: held-out is {100*ratio:.1f}% of pool — outside the 20-30% target"
        )


def test_categories_match_on_both_sides():
    assert set(TRAIN_PROMPTS) == set(HELD_OUT_PROMPTS) == set(CATEGORIES)


def test_no_duplicate_prompts_within_train_categories():
    """A prompt should not appear in two categories of the train pool."""
    seen: dict[str, str] = {}
    for cat, prompts in TRAIN_PROMPTS.items():
        for p in prompts:
            if p in seen:
                pytest.fail(f"prompt {p!r} in both {seen[p]} and {cat}")
            seen[p] = cat


def test_no_duplicate_prompts_within_held_out_categories():
    seen: dict[str, str] = {}
    for cat, prompts in HELD_OUT_PROMPTS.items():
        for p in prompts:
            if p in seen:
                pytest.fail(f"prompt {p!r} in both {seen[p]} and {cat}")
            seen[p] = cat


def test_eval_prompts_alias():
    """EVAL_PROMPTS is an alias for HELD_OUT_PROMPTS — old callers keep working."""
    assert EVAL_PROMPTS is HELD_OUT_PROMPTS
    assert ALL_EVAL_PROMPTS is ALL_HELD_OUT_PROMPTS
    assert get_eval_prompts() is HELD_OUT_PROMPTS


def test_sample_train_prompts_returns_only_train_prompts():
    """Sanity: training never sees a held-out prompt."""
    import random
    rng = random.Random(42)
    samples = sample_train_prompts(50, rng=rng)
    held_out_set = set(ALL_HELD_OUT_PROMPTS)
    leaked = [p for p in samples if p in held_out_set]
    assert not leaked, f"sample_train_prompts leaked held-out prompts: {leaked}"


def test_sample_train_prompts_size():
    samples = sample_train_prompts(7)
    assert len(samples) == 7


def test_sample_train_prompts_deterministic_with_rng():
    import random
    a = sample_train_prompts(10, rng=random.Random(0))
    b = sample_train_prompts(10, rng=random.Random(0))
    assert a == b
