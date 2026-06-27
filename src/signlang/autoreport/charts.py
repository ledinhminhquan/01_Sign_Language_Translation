"""Matplotlib charts for the sign-translation report/slides.

  * a **recognition** bar chart - agent gloss accuracy vs most-frequent / random baselines;
  * a **quality** chart - gloss accuracy + BLEU + chrF + segmentation-F1;
  * an **outcome bucket** chart (exact / abstained / wrong) from error analysis.

Returns saved PNG paths under ``run_dir()/report``; matplotlib lazy-imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger
from . import artifact_loader as AL

logger = get_logger(__name__)

_AGENT = "#2b6cb0"
_BASE = "#9aa7b4"
_GOOD = "#2f855a"
_MED = "#dd6b20"
_POOR = "#c53030"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def recognition_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not AL.has_eval(arts):
        return None
    agent = (AL.agent_recog(arts, "gloss_accuracy") or 0.0) * 100
    mf = (AL.baseline_recog(arts, "most_frequent_gloss", "gloss_accuracy") or 0.0) * 100
    rnd = (AL.baseline_recog(arts, "random_gloss", "gloss_accuracy") or 0.0) * 100
    try:
        plt = _mpl()
        labels = ["agent\n(trained core)", "most-frequent", "random"]
        vals = [agent, mf, rnd]
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.bar(labels, vals, color=[_AGENT, _BASE, "#cbd5e0"])
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("gloss accuracy %")
        ax.set_title("Sign recognition: trained core vs baselines")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("recognition_chart skipped (%s)", exc)
        return None


def quality_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    acc = AL.agent_recog(arts, "gloss_accuracy")
    bleu = AL.agent_trans(arts, "bleu")
    chrf = AL.agent_trans(arts, "chrf")
    sf = AL.seg_f1(arts)
    if acc is None and bleu is None:
        return None
    try:
        plt = _mpl()
        labels = ["gloss acc", "BLEU", "chrF", "seg-F1"]
        vals = [(acc or 0.0) * 100, bleu or 0.0, chrf or 0.0, (sf or 0.0) * 100]
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.bar(labels, vals, color=[_AGENT, _GOOD, "#553c9a", "#2b6cb0"])
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("score")
        ax.set_title("Recognition + translation + segmentation quality")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("quality_chart skipped (%s)", exc)
        return None


def buckets_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    b = AL.buckets(arts)
    vals = [b.get("correct"), b.get("abstained"), b.get("wrong")]
    if not any(isinstance(v, (int, float)) for v in vals):
        return None
    try:
        plt = _mpl()
        labels = ["exact", "abstained", "wrong"]
        nums = [float(v) if isinstance(v, (int, float)) else 0.0 for v in vals]
        fig, ax = plt.subplots(figsize=(5.6, 3.3))
        ax.bar(labels, nums, color=[_GOOD, _MED, _POOR])
        for i, v in enumerate(nums):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("# sentences"); ax.set_title("Agent outcomes (exact / abstained / wrong)")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("buckets_chart skipped (%s)", exc)
        return None


def build_all(arts: Dict[str, Any], out_dir: Path) -> List[Tuple[str, Path]]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return []
    charts: List[Tuple[str, Path]] = []
    jobs = [("recognition", lambda p: recognition_chart(arts, p)),
            ("quality", lambda p: quality_chart(arts, p)),
            ("buckets", lambda p: buckets_chart(arts, p))]
    for name, fn in jobs:
        try:
            p = fn(out_dir / f"{name}.png")
        except Exception as exc:
            logger.info("chart %s skipped (%s)", name, exc)
            p = None
        if p:
            charts.append((name, p))
    return charts


__all__ = ["recognition_chart", "quality_chart", "buckets_chart", "build_all"]
