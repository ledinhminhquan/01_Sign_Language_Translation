"""Persist the offline centroid recognizer + record the trivial baselines.

The numpy nearest-centroid gloss recognizer needs no GPU; this builder fits it on the synthetic
collection, saves it under the model dir, and records the most-frequent / random / identity
baselines so the registry, report and API can reference the floors the trained core must beat.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models import baseline as B
from ..models.sign2text import fit_centroid_recognizer

logger = get_logger(__name__)


def build_baseline(cfg: AppConfig, limit: Optional[int] = None) -> Dict:
    rec = fit_centroid_recognizer(cfg, n_train=limit)
    rec_dir = cfg.model.output_dir / "centroid"
    rec.save(rec_dir)
    payload = {"name": "baselines", "version": "baseline-1.0",
               "centroid_recognizer": str(rec_dir),
               "most_frequent_gloss": B.most_frequent_gloss(cfg),
               "random_gloss": "uniform over the vocabulary",
               "identity_translate": "gloss tokens lowercased as text (no lexicon)"}
    out_path = cfg.model.output_dir / cfg.model.baseline_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("baselines -> %s (centroid -> %s)", out_path, rec_dir)
    return {"baseline_path": str(out_path), "centroid_dir": str(rec_dir)}


__all__ = ["build_baseline"]
