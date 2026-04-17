"""
Train and eval prompt sets.

Eval prompts are stratified into three categories by expected diversity ceiling:
  - constrained:     single object + specific attributes → low diversity ceiling
  - common_subject:  recognisable but flexible → medium ceiling
  - open_ended:      abstract / creative → high ceiling (collapse signal strongest)
"""

from __future__ import annotations

import random
from typing import Dict, List

# ── Training prompts ─────────────────────────────────────────────────────────
# A broad pool of prompts that the RL training loop samples from each iteration.

TRAIN_PROMPTS: List[str] = [
    "a photo of a cat sitting on a windowsill",
    "a beautiful sunset over the ocean",
    "a portrait of a young woman smiling",
    "a futuristic cityscape at night",
    "a bowl of fresh fruit on a wooden table",
    "a golden retriever playing in the park",
    "a mountain landscape with a lake",
    "a cozy cabin in the snowy woods",
    "a street scene in Paris with the Eiffel Tower",
    "a colorful bouquet of wildflowers",
    "a photo of a red sports car on a highway",
    "a fantasy castle on a floating island",
    "an astronaut floating in space with Earth in the background",
    "a tropical beach with palm trees and turquoise water",
    "a plate of sushi on a black stone plate",
    "a rainy city street with neon reflections",
    "a white horse running through a field of flowers",
    "a steampunk robot reading a book",
    "an oil painting of a medieval village",
    "a photorealistic portrait of an elderly man with kind eyes",
    "a watercolor painting of cherry blossoms",
    "a drone shot of a coral reef",
    "a macro photo of a butterfly on a flower",
    "a minimalist modern living room interior",
    "a cyberpunk street market at dusk",
    "a van Gogh style painting of a starry sky over mountains",
    "a photo of a lighthouse on a stormy coast",
    "a surreal melting clock landscape",
    "a rustic Italian countryside vineyard",
    "a photo of a baby panda eating bamboo",
    "a dark fantasy forest with glowing mushrooms",
    "a vintage photo of a 1950s diner",
]


# ── Evaluation prompts (held-out, stratified) ───────────────────────────────

EVAL_PROMPTS: Dict[str, List[str]] = {
    "constrained": [
        "a single red apple on a white table",
        "a black cat with green eyes sitting on a grey couch",
        "a blue coffee mug on a wooden desk next to a laptop",
        "a white ceramic vase with three yellow tulips",
        "a brown leather briefcase on a marble floor",
    ],
    "common_subject": [
        "a dog playing fetch in a sunny park",
        "a chef preparing food in a restaurant kitchen",
        "a child building a sandcastle on the beach",
        "a musician playing guitar on a street corner",
        "birds flying over a lake at sunrise",
        "a couple walking through an autumn forest",
        "a busy farmer's market with colorful produce",
    ],
    "open_ended": [
        "a dreamlike landscape",
        "the feeling of nostalgia",
        "an abstract representation of joy",
        "a world where gravity works differently",
        "the intersection of nature and technology",
        "a scene from a civilization a thousand years in the future",
        "the concept of time passing",
        "an impossible architecture that defies physics",
    ],
}

# Flat list for iteration convenience
ALL_EVAL_PROMPTS: List[str] = []
for _cat_prompts in EVAL_PROMPTS.values():
    ALL_EVAL_PROMPTS.extend(_cat_prompts)


def sample_train_prompts(n: int, rng: random.Random | None = None) -> List[str]:
    """Sample n prompts from the training pool (with replacement)."""
    if rng is None:
        return random.choices(TRAIN_PROMPTS, k=n)
    return rng.choices(TRAIN_PROMPTS, k=n)


def get_eval_prompts() -> Dict[str, List[str]]:
    """Return the structured eval prompt dict."""
    return EVAL_PROMPTS
