"""Pure decision helpers for the sign-translation agent (no I/O, easy to unit-test).

Each function returns a (branch, detail) pair the agent records as a Decision (D1-D5).
"""

from __future__ import annotations

from typing import List, Tuple

from ..config import AgentConfig


def ingest_gate(n_frames: int, cfg: AgentConfig) -> Tuple[str, str]:
    """D1 - reject too-short / empty sequences."""
    if n_frames < cfg.min_frames:
        return "fail", f"only {n_frames} frames (< {cfg.min_frames})"
    return "proceed", f"{n_frames} frames"


def segment_branch(spans: List[Tuple[int, int]]) -> Tuple[str, str]:
    """D2 - did segmentation find discrete sign units?"""
    if len(spans) <= 1:
        return "single_span", f"{len(spans)} span (no clear sign boundaries)"
    return "multi", f"{len(spans)} sign segments"


def confidence_gate(confs: List[float], cfg: AgentConfig) -> Tuple[str, str, int]:
    """D3 - count low-confidence gloss predictions."""
    n_low = sum(1 for c in confs if c < cfg.recog_min_conf)
    if n_low == 0:
        return "confident", "all glosses above threshold", 0
    return "low_confidence", f"{n_low}/{len(confs)} glosses below {cfg.recog_min_conf}", n_low


def verify_branch(chrf_estimate: float, cfg: AgentConfig) -> Tuple[str, str]:
    """D4 - keep the translation if it clears the chrF floor (when a round-trip check is available)."""
    if not cfg.verify_backtranslate:
        return "kept", "verification disabled"
    if chrf_estimate >= cfg.min_chrf_keep:
        return "kept", f"round-trip chrF {chrf_estimate:.1f} >= {cfg.min_chrf_keep}"
    return "flagged", f"round-trip chrF {chrf_estimate:.1f} < {cfg.min_chrf_keep}"


def abstain_gate(n_segments: int, n_low_conf: int, cfg: AgentConfig) -> Tuple[bool, str]:
    """D5 - abstain when too much of the utterance is OOV / low-confidence."""
    if not cfg.abstain_enabled or n_segments == 0:
        return False, "abstain disabled or nothing to judge"
    ratio = n_low_conf / n_segments
    if ratio > cfg.oov_abstain_ratio:
        return True, f"{n_low_conf}/{n_segments} segments low-confidence (> {cfg.oov_abstain_ratio:.0%})"
    return False, f"{n_low_conf}/{n_segments} segments low-confidence (within tolerance)"


__all__ = ["ingest_gate", "segment_branch", "confidence_gate", "verify_branch", "abstain_gate"]
