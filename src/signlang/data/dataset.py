"""Data loading: synthetic pose sentences (default) + an optional permissive real smoke-test corpus.

``datasets`` is imported lazily; everything falls back to the synthetic generator offline. Real
sign-language corpora are restrictively licensed (see docs/data_card.md) - only
``Sigurdur/icelandic-sign-language`` (Apache-2.0) is loaded here, as a loader smoke-test, never as
the main benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig, data_dir
from ..logging_utils import get_logger
from . import samples

logger = get_logger(__name__)


def load_training(cfg: AppConfig, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The synthetic training collection (the spine)."""
    n = limit or cfg.data.n_sentences
    return samples.seed_collection(cfg, n=n)


def load_real_smoke(cfg: AppConfig, limit: int = 8) -> Dict[str, Any]:
    """Probe the only cleanly-permissive real corpus (Apache-2.0); returns a small preview."""
    try:
        from datasets import load_dataset  # lazy
        ds = load_dataset(cfg.data.real_smoke_dataset, split="train", streaming=True)
        rows = []
        for i, r in enumerate(ds):
            if i >= limit:
                break
            rows.append({k: (str(v)[:80] if not isinstance(v, (int, float)) else v) for k, v in r.items()})
        return {"ok": True, "dataset": cfg.data.real_smoke_dataset, "license": "apache-2.0", "preview": rows}
    except Exception as exc:
        logger.warning("real smoke corpus probe failed (%s)", exc)
        return {"ok": False, "dataset": cfg.data.real_smoke_dataset, "error": str(exc)}


def synthetic_dir(split: str = "eval") -> Path:
    return data_dir() / "synthetic" / split


def build_synthetic(cfg: AppConfig, n: Optional[int] = None, split: str = "eval") -> Dict:
    from .synth_pose import generate_collection
    return generate_collection(synthetic_dir(split), n=n or cfg.data.n_sentences, cfg=cfg, seed=cfg.data.seed)


__all__ = ["load_training", "load_real_smoke", "synthetic_dir", "build_synthetic"]
