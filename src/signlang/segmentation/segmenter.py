"""Motion-based sign segmentation - split a continuous pose sequence into sign units.

Signs are separated by low-motion REST/transition frames. We compute the per-frame velocity
magnitude, threshold it at a low quantile to mark "still" frames, and cut the sequence into
high-motion runs (the candidate signs). Pure algorithmic (no training); the recognizer then labels
each segment. When the synthetic gold spec carries boundaries we can score this against them.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..config import AgentConfig
from ..logging_utils import get_logger
from ..data.synth_pose import PoseSequence, motion_profile

logger = get_logger(__name__)


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(q * (len(s) - 1))))
    return s[i]


def segment(seq: PoseSequence, cfg: AgentConfig) -> List[Tuple[int, int]]:
    """Return [(start, end)] frame spans for the candidate signs (high-motion runs)."""
    vel = motion_profile(seq.frames)
    n = len(vel)
    if n < cfg.min_segment_frames:
        return [(0, n)] if n else []
    pos = [v for v in vel if v > 0] or vel
    thr = 0.5 * (_quantile(pos, 0.9) + _quantile(pos, cfg.motion_quantile))   # between rest noise + sign motion
    active = [v > thr for v in vel]
    # A sign is one high-motion run; only a SUSTAINED low-motion gap (>= min_gap frames, i.e. a real
    # rest between signs) closes a segment - a 1-frame velocity dip inside a sign does not split it.
    min_gap = max(2, cfg.min_segment_frames // 2)
    spans: List[Tuple[int, int]] = []
    start = None
    gap = 0
    for t, a in enumerate(active):
        if a:
            if start is None:
                start = t
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_gap:
                    end = t - gap + 1
                    if end - start >= cfg.min_segment_frames:
                        spans.append((start, end))
                    start = None
                    gap = 0
    if start is not None and n - start >= cfg.min_segment_frames:
        spans.append((start, n))
    if not spans:                      # fall back to the whole sequence
        spans = [(0, n)]
    return spans


def boundary_f1(pred: List[Tuple[int, int]], gold: List[Tuple[int, int]], tol: int = 3) -> dict:
    """Match predicted sign spans to gold by start-frame proximity (within tol)."""
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": len(pred), "n_gold": 0}
    used = set()
    tp = 0
    for ps, _ in pred:
        for gi, (gs, _) in enumerate(gold):
            if gi in used:
                continue
            if abs(ps - gs) <= tol:
                tp += 1
                used.add(gi)
                break
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "n_pred": len(pred), "n_gold": len(gold)}


__all__ = ["segment", "boundary_f1"]
