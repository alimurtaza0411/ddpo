"""Tests for CLIPScoreFaithfulness (Phase 2 task 3).

Uses fake CLIP model + processor injected via the `_model=`, `_processor=`
seam — no real CLIP weights are loaded in these tests.
"""

from __future__ import annotations

import pytest
import torch

from ddpo.rewards import CLIPScoreFaithfulness
from ddpo.reward_ensemble import REWARD_REGISTRY


# ── Fakes ────────────────────────────────────────────────────────────────────

class _FakeBatch(dict):
    """Behaves like CLIPProcessor's BatchFeature: subscript + .to(...)."""
    def to(self, *args, **kwargs):
        return self


class _FakeProcessor:
    """Returns canned tensors regardless of inputs.

    Each `.score` call passes through here twice (once for images, once for
    text). We return shape-correct fake tensors so the wrapper can drive its
    cosine math.
    """
    def __init__(self, image_features, text_features):
        self.image_features = image_features
        self.text_features = text_features

    def __call__(self, **kwargs):
        # Branch on which features the caller wants
        if "images" in kwargs:
            B = len(kwargs["images"])
            return _FakeBatch({"_branch": "images", "_B": B})
        if "text" in kwargs:
            B = len(kwargs["text"])
            return _FakeBatch({"_branch": "text", "_B": B})
        raise ValueError("processor called without images or text")


class _FakeCLIPModel:
    """Returns the canned image/text features irrespective of input contents."""
    def __init__(self, image_features, text_features):
        self.image_features = image_features
        self.text_features = text_features

    def get_image_features(self, **inputs):
        return self.image_features

    def get_text_features(self, **inputs):
        return self.text_features


def _make_score(image_feats, text_feats):
    """Helper: build a CLIPScoreFaithfulness with injected fakes."""
    return CLIPScoreFaithfulness(
        device="cpu",
        dtype=torch.float32,
        _model=_FakeCLIPModel(image_feats, text_feats),
        _processor=_FakeProcessor(image_feats, text_feats),
    )


# ── Cosine-similarity correctness ────────────────────────────────────────────

def test_identical_embeddings_give_cosine_one():
    feats = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
    score_fn = _make_score(feats, feats).score
    out = score_fn(images=["i", "i"], prompts=["p", "p"])
    assert torch.allclose(out, torch.ones(2), atol=1e-5)


def test_orthogonal_embeddings_give_cosine_zero():
    img = torch.tensor([[1.0, 0.0]])
    txt = torch.tensor([[0.0, 1.0]])
    score_fn = _make_score(img, txt).score
    out = score_fn(images=["i"], prompts=["p"])
    assert torch.allclose(out, torch.zeros(1), atol=1e-5)


def test_anti_aligned_embeddings_give_cosine_minus_one():
    img = torch.tensor([[1.0, 0.0]])
    txt = torch.tensor([[-1.0, 0.0]])
    score_fn = _make_score(img, txt).score
    out = score_fn(images=["i"], prompts=["p"])
    assert torch.allclose(out, torch.tensor([-1.0]), atol=1e-5)


def test_norms_do_not_affect_score():
    """Cosine is scale-invariant: scaling either side leaves the score unchanged."""
    img = torch.tensor([[3.0, 4.0]])             # norm 5
    txt = torch.tensor([[600.0, 800.0]])         # norm 1000, same direction
    score_fn = _make_score(img, txt).score
    out = score_fn(images=["i"], prompts=["p"])
    assert torch.allclose(out, torch.ones(1), atol=1e-5)


def test_handles_basemodel_output_with_pooling_wrapper():
    """transformers >=5 sometimes returns a wrapper instead of a tensor."""
    class _Wrapped:
        def __init__(self, t):
            self.image_embeds = t
            self.text_embeds = t

    img_t = torch.tensor([[1.0, 0.0]])
    txt_t = torch.tensor([[1.0, 0.0]])

    # Inject wrappers
    class _ModelW:
        def __init__(self, img, txt):
            self.img = _Wrapped(img); self.txt = _Wrapped(txt)
        def get_image_features(self, **kw): return self.img
        def get_text_features(self, **kw): return self.txt

    fake = CLIPScoreFaithfulness(
        device="cpu", dtype=torch.float32,
        _model=_ModelW(img_t, txt_t),
        _processor=_FakeProcessor(img_t, txt_t),
    )
    out = fake.score(images=["i"], prompts=["p"])
    assert torch.allclose(out, torch.ones(1), atol=1e-5)


def test_zero_embedding_does_not_nan():
    """clamp_min(1e-8) on the norm should keep things finite even for zero vectors."""
    img = torch.zeros(1, 4)
    txt = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    score_fn = _make_score(img, txt).score
    out = score_fn(images=["i"], prompts=["p"])
    assert torch.isfinite(out).all(), f"Got non-finite score: {out}"


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_returns_cpu_float32_tensor():
    feats = torch.tensor([[1.0, 0.0]] * 3)
    out = _make_score(feats, feats).score(["i"] * 3, ["p"] * 3)
    assert out.device.type == "cpu"
    assert out.dtype == torch.float32
    assert out.shape == (3,)


def test_mismatched_lengths_raise():
    feats = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="must be the same length"):
        _make_score(feats, feats).score(images=["a", "b"], prompts=["x"])


def test_empty_inputs_return_empty_tensor():
    feats = torch.tensor([[1.0, 0.0]])
    out = _make_score(feats, feats).score(images=[], prompts=[])
    assert out.shape == (0,)


def test_clipscore_registered():
    assert "clipscore" in REWARD_REGISTRY


def test_plays_well_with_ensemble():
    img = torch.tensor([[1.0, 0.0]])
    txt = torch.tensor([[1.0, 0.0]])
    cs = _make_score(img, txt).score

    from ddpo.reward_ensemble import RewardEnsemble
    ens = RewardEnsemble([(cs, 0.7, "clip")])
    combined, per = ens(images=["i"], prompts=["p"])
    assert torch.allclose(combined, torch.tensor([0.7]), atol=1e-5)  # 1.0 * 0.7
    assert torch.allclose(per["clip"], torch.tensor([1.0]), atol=1e-5)
