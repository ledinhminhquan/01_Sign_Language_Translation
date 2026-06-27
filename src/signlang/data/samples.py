"""Seed collection + held-out eval examples for the offline pipeline (no torch/video/network)."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import AppConfig
from .synth_pose import make_sentence


def seed_collection(cfg: AppConfig = None, n: int = 40) -> List[Dict[str, Any]]:
    """A small set of synthetic sentences with gold glosses + text (for demo/quality)."""
    cfg = cfg or AppConfig()
    out: List[Dict[str, Any]] = []
    for i in range(n):
        seq = make_sentence(cfg.data.seed + i, cfg)
        out.append({"id": f"seq_{i:04d}", "seq": seq, "glosses": seq.spec["glosses"],
                    "text": seq.spec["text"], "boundaries": seq.spec["boundaries"]})
    return out


def seed_eval(cfg: AppConfig = None, n: int = 80) -> List[Dict[str, Any]]:
    """A held-out split (different seed range) for evaluation."""
    cfg = cfg or AppConfig()
    out: List[Dict[str, Any]] = []
    for i in range(n):
        seq = make_sentence(cfg.data.seed + 5000 + i, cfg)
        out.append({"id": f"eval_{i:04d}", "seq": seq, "glosses": seq.spec["glosses"],
                    "text": seq.spec["text"], "boundaries": seq.spec["boundaries"]})
    return out


__all__ = ["seed_collection", "seed_eval"]
