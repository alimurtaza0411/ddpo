"""
Multi-objective reward composition for DDPO.

Wraps any number of per-sample reward callables `(images, prompts) -> (B,) tensor`
into a single weighted sum, while still exposing per-component scores for logging.

Math (per training iter):
  r_combined = Σᵢ λᵢ · rᵢ(images, prompts)

The combined reward feeds the PPO advantage; per-component rewards are returned
alongside so the caller can log them and watch each objective independently.

Usage:

    ens = RewardEnsemble([
        (pickscore_fn, 1.0, "pickscore"),
        (clipscore_fn, 0.5, "clipscore"),
        (diversity_fn, 0.3, "diversity"),
    ])
    combined, per_comp = ens(images, prompts)

A bare callable is auto-wrapped (weight 1.0, name "reward") so legacy code that
passed a single reward_fn keeps working without changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

import torch

# A single reward component: (callable, weight, name).
RewardCallable = Callable[[Sequence, Sequence[str]], torch.Tensor]
RewardSpec = Tuple[RewardCallable, float, str]


@dataclass
class _Component:
    fn: RewardCallable
    weight: float
    name: str


class RewardEnsemble:
    """
    Weighted ensemble of reward functions.

    Each component is `(callable, weight, name)`. `__call__(images, prompts)`
    returns `(combined_reward, per_component_dict)` where:
      - combined_reward: (B,) tensor = Σᵢ wᵢ · componentᵢ(images, prompts)
      - per_component_dict: name -> (B,) tensor of UNWEIGHTED scores
        (useful for logging the raw signal independent of the chosen weight)
    """

    def __init__(self, components: Sequence[Union[RewardSpec, _Component]]):
        if not components:
            raise ValueError("RewardEnsemble needs at least one component")
        self.components: List[_Component] = []
        for c in components:
            if isinstance(c, _Component):
                self.components.append(c)
            else:
                fn, w, name = c
                self.components.append(_Component(fn=fn, weight=float(w), name=name))
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate component names in ensemble: {names}")

    def __len__(self) -> int:
        return len(self.components)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.components]

    @property
    def weights(self) -> List[float]:
        return [c.weight for c in self.components]

    def __call__(
        self,
        images: Sequence,
        prompts: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        per: Dict[str, torch.Tensor] = {}
        combined: torch.Tensor | None = None
        for c in self.components:
            r = c.fn(images, prompts)
            if not torch.is_tensor(r):
                r = torch.as_tensor(r)
            r = r.detach().float().cpu()
            per[c.name] = r
            contribution = c.weight * r
            combined = contribution if combined is None else combined + contribution
        assert combined is not None  # guarded by __init__
        return combined, per


def coerce_reward(
    reward: Union[RewardCallable, RewardEnsemble],
) -> RewardEnsemble:
    """
    Promote a bare callable to a single-component ensemble.
    Returns the input unchanged if it's already an ensemble.
    """
    if isinstance(reward, RewardEnsemble):
        return reward
    if callable(reward):
        return RewardEnsemble([(reward, 1.0, "reward")])
    raise TypeError(f"Expected callable or RewardEnsemble, got {type(reward)!r}")


# ── Registry & config-driven construction ────────────────────────────────────

# A factory turns a per-component config dict into a reward callable.
RewardFactory = Callable[[Mapping[str, Any]], RewardCallable]
REWARD_REGISTRY: Dict[str, RewardFactory] = {}


def register_reward(name: str):
    """Decorator: register a factory under `name` so configs can reference it."""

    def deco(factory: RewardFactory) -> RewardFactory:
        if name in REWARD_REGISTRY:
            raise ValueError(f"Reward type {name!r} already registered")
        REWARD_REGISTRY[name] = factory
        return factory

    return deco


def build_ensemble(cfg_list: Sequence[Mapping[str, Any]]) -> RewardEnsemble:
    """
    Build a RewardEnsemble from a list-of-dicts config block.

    Each dict must have:
      - `type`: a key in REWARD_REGISTRY
      - `weight`: float (default 1.0)
      - `name`: optional, defaults to `type`
      - `config`: optional dict passed to the factory
    """
    components: List[RewardSpec] = []
    for entry in cfg_list:
        rtype = entry["type"]
        if rtype not in REWARD_REGISTRY:
            raise KeyError(
                f"Reward type {rtype!r} not registered. "
                f"Known: {sorted(REWARD_REGISTRY)}"
            )
        weight = float(entry.get("weight", 1.0))
        name = entry.get("name", rtype)
        sub_cfg = entry.get("config", {}) or {}
        fn = REWARD_REGISTRY[rtype](sub_cfg)
        components.append((fn, weight, name))
    return RewardEnsemble(components)
