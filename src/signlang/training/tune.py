"""Pose-noise robustness sweep - how recognition + translation degrade as keypoints get noisier.

Sweeps the injected pose-keypoint noise and reports gloss accuracy + BLEU + the abstain rate on a
held-out synthetic split - a cheap proxy for how the segmenter + recognizer + abstention hold up as
the pose front-end (MediaPipe on low-quality video) gets noisier.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger
from .evaluate import evaluate

logger = get_logger(__name__)


def tune(cfg: AppConfig, noises: Optional[List[float]] = None, save: bool = True,
         load_model: bool = True) -> Dict:
    noises = noises or [0.0, 0.1, 0.25, 0.5]
    trials: List[Dict[str, Any]] = []
    for noise in noises:
        rep = evaluate(cfg, limit=60, load_model=load_model, pose_noise=noise, save=False)
        h = rep["headline"]
        trials.append({"pose_noise": noise, "gloss_accuracy": h["gloss_accuracy"], "bleu": h["bleu"],
                       "segmentation_f1": h["segmentation_f1"], "abstain_rate": rep["abstain_rate"]})
    best = max(trials, key=lambda t: t["gloss_accuracy"]) if trials else {}
    result = {"trials": trials, "best": best}
    if save:
        out = run_dir() / "tune"
        out.mkdir(parents=True, exist_ok=True)
        (out / "tune.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("tune: best noise=%s gloss_acc=%s", best.get("pose_noise"), best.get("gloss_accuracy"))
    return result


__all__ = ["tune"]
