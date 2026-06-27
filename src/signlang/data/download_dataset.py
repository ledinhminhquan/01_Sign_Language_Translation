"""Prefetch + sanity-check: probe the seq2seq backbone + the permissive real corpus, report the
synthetic seed stats, and optionally render the synthetic collection. Degrades gracefully.
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _probe(loader) -> Dict[str, Any]:
    try:
        return {"ok": True, **loader()}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


def download_all(cfg: AppConfig, render_synthetic: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"backbone": {}, "real_smoke": {}, "seed": {}, "synthetic": {}}

    def backbone_probe():
        from transformers import AutoTokenizer  # noqa: F401
        AutoTokenizer.from_pretrained(cfg.model.text_backbone)
        return {"model": cfg.model.text_backbone, "reachable": True}

    out["backbone"] = _probe(backbone_probe)
    if cfg.data.use_hf:
        from .dataset import load_real_smoke
        out["real_smoke"] = load_real_smoke(cfg)
    else:
        out["real_smoke"] = {"ok": True, "note": "use_hf off; synthetic pose generator is the primary data "
                             "(real continuous-SLT corpora are NC/gated - see docs/data_card.md)"}

    from . import samples
    coll = samples.seed_collection(cfg, n=12)
    out["seed"] = {"ok": True, "sentences": len(coll), "keypoint_dim": cfg.pose.keypoint_dim,
                   "example_glosses": coll[0]["glosses"], "example_text": coll[0]["text"]}

    if render_synthetic:
        try:
            from .dataset import build_synthetic
            out["synthetic"] = {"ok": True, **build_synthetic(cfg)}
        except Exception as exc:
            out["synthetic"] = {"ok": False, "error": str(exc)}

    logger.info("download_all: backbone=%s seed=%d sentences", out["backbone"].get("ok"), out["seed"]["sentences"])
    return out


__all__ = ["download_all"]
