"""Baselines for recognition + translation - the floors the trained core must beat.

Recognition: ``MostFrequentRecognizer`` (always the most common gloss), ``RandomRecognizer``.
Translation: ``identity_translate`` (use the gloss tokens lowercased as the text - measures the
value of the gloss->text lexicon/translation).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from ..config import AppConfig
from ..data import lexicon
from ..data.synth_pose import PoseSequence, make_sentence


class MostFrequentRecognizer:
    name = "most_frequent"
    version = "baseline"

    def __init__(self, gloss: str):
        self.gloss = gloss

    def recognize(self, seq: PoseSequence, spans: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
        return [(self.gloss, 0.0) for _ in spans]


class RandomRecognizer:
    name = "random"
    version = "baseline"

    def __init__(self, vocab: List[str], seed: int = 0):
        self.vocab = vocab
        self._rng = random.Random(seed)

    def recognize(self, seq: PoseSequence, spans: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
        return [(self._rng.choice(self.vocab), 0.0) for _ in spans]


def most_frequent_gloss(cfg: AppConfig, n: int = 200) -> str:
    counts = {}
    for i in range(min(n, cfg.data.n_sentences)):
        for g in make_sentence(cfg.data.seed + i, cfg).spec["glosses"]:
            counts[g] = counts.get(g, 0) + 1
    return max(counts, key=counts.get) if counts else lexicon.vocab(cfg.data.vocab_size)[0]


def build_most_frequent(cfg: AppConfig) -> MostFrequentRecognizer:
    return MostFrequentRecognizer(most_frequent_gloss(cfg))


def build_random(cfg: AppConfig) -> RandomRecognizer:
    return RandomRecognizer(lexicon.vocab(cfg.data.vocab_size), seed=cfg.data.seed)


def identity_translate(glosses: List[str]) -> str:
    """No lexicon: just lowercase the gloss tokens (e.g. THANK-YOU -> 'thank-you')."""
    return " ".join(g.lower() for g in glosses)


__all__ = ["MostFrequentRecognizer", "RandomRecognizer", "most_frequent_gloss",
           "build_most_frequent", "build_random", "identity_translate"]
