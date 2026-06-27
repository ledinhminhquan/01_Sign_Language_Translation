"""Translation-quality report (the P01 special quality analysis).

Runs the agent on the held-out synthetic split and reports recognition (gloss WER/accuracy) +
translation (BLEU/chrF/WER) + segmentation boundary-F1 + the abstain rate - the headline quality
picture, alongside the most-frequent / identity baselines.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def quality_report(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
    cfg = cfg or AppConfig()
    try:
        from ..training.evaluate import evaluate
        rep = evaluate(cfg, limit=limit or 80, load_model=False, save=False)
    except Exception as exc:
        result = {"error": str(exc)}
        if save:
            _save(result)
        return result
    sys = rep["systems"]
    result = {"n": rep["n"],
              "gloss_accuracy": sys["agent"]["recognition"]["gloss_accuracy"],
              "gloss_wer": sys["agent"]["recognition"]["gloss_wer"],
              "bleu": sys["agent"]["translation"]["bleu"],
              "chrf": sys["agent"]["translation"]["chrf"],
              "segmentation_boundary_f1": rep["segmentation_boundary_f1"],
              "abstain_rate": rep["abstain_rate"],
              "most_frequent_gloss_accuracy": sys["most_frequent_gloss"]["recognition"]["gloss_accuracy"],
              "identity_translate_bleu": sys["identity_translate"]["translation"]["bleu"]}
    if save:
        _save(result)
    logger.info("quality: gloss_acc=%s BLEU=%s seg_f1=%s", result["gloss_accuracy"], result["bleu"],
                result["segmentation_boundary_f1"])
    return result


def _save(out: Dict) -> None:
    try:
        d = run_dir() / "quality"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"quality-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.info("quality: could not save (%s)", exc)


__all__ = ["quality_report"]
