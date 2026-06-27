"""The five agent tools (D1-D5). Each mutates the JobState and records a Decision.

D1 ingest -> D2 segment -> D3 recognize -> D4 translate+verify -> D5 finalize/abstain.
The recognizer + translator are injected (offline: numpy centroid recognizer + lexicon translator;
Colab: the trained transformer recognizer + t5 translator).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config import AppConfig
from ..data import lexicon
from ..segmentation.segmenter import segment as motion_segment
from . import policy
from .state import JobState, JobStatus


def tool_ingest(job: JobState, seq, cfg: AppConfig) -> JobState:
    job.n_frames = getattr(seq, "n_frames", 0) or 0
    job.source = "pose"
    branch, detail = policy.ingest_gate(job.n_frames, cfg.agent)
    job.add_decision("D1", "ingest_gate", branch, detail)
    if branch == "fail":
        job.status = JobStatus.FAILED
    else:
        job.status = JobStatus.INGESTED
    return job


def tool_segment(job: JobState, seq, cfg: AppConfig) -> JobState:
    spans = motion_segment(seq, cfg.agent)
    job.segments = spans
    branch, detail = policy.segment_branch(spans)
    job.add_decision("D2", "segmentation", branch, detail)
    job.status = JobStatus.SEGMENTED
    return job


def tool_recognize(job: JobState, seq, cfg: AppConfig, recognizer) -> JobState:
    preds: List[Tuple[str, float]] = recognizer.recognize(seq, job.segments)
    job.glosses = [g for g, _ in preds]
    job.gloss_confs = [float(c) for _, c in preds]
    job.mean_conf = sum(job.gloss_confs) / len(job.gloss_confs) if job.gloss_confs else 0.0
    branch, detail, n_low = policy.confidence_gate(job.gloss_confs, cfg.agent)
    job.n_low_conf = n_low
    job.low_confidence = branch == "low_confidence"
    job.add_decision("D3", "confidence_gate", branch, detail)
    job.model_versions["recognizer"] = getattr(recognizer, "version", getattr(recognizer, "name", "?"))
    job.status = JobStatus.RECOGNIZED
    return job


def _roundtrip_agreement(glosses: List[str], text: str, cfg: AppConfig) -> float:
    """Reverse-map the translated text back to glosses via the inverse lexicon and measure agreement."""
    lex = lexicon.gloss_to_text(cfg.data.vocab_size)
    inv: Dict[str, str] = {}
    for g, t in lex.items():
        inv[t] = g
        for w in t.split():
            inv.setdefault(w, g)
    words = text.split()
    recovered: List[str] = []
    i = 0
    while i < len(words):
        two = " ".join(words[i:i + 2])
        if two in inv:
            recovered.append(inv[two]); i += 2
        else:
            recovered.append(inv.get(words[i], words[i].upper())); i += 1
    if not glosses:
        return 1.0
    match = sum(1 for a, b in zip(glosses, recovered) if a == b)
    return match / max(len(glosses), len(recovered))


def tool_translate(job: JobState, cfg: AppConfig, translator=None) -> JobState:
    from ..models.sign2text import translate as do_translate
    job.text = do_translate(job.glosses, cfg, model=translator)
    agreement = _roundtrip_agreement(job.glosses, job.text, cfg)
    job.metrics["roundtrip_agreement"] = round(agreement, 4)
    branch, detail = policy.verify_branch(agreement, cfg.agent)
    job.add_decision("D4", "translate_verify", branch, detail)
    if branch == "flagged":
        job.needs_review = True
    job.model_versions["translator"] = getattr(translator, "version", "lexicon") if translator else "lexicon"
    job.status = JobStatus.TRANSLATED
    return job


def tool_finalize(job: JobState, cfg: AppConfig) -> JobState:
    abstain, detail = policy.abstain_gate(len(job.segments), job.n_low_conf, cfg.agent)
    job.add_decision("D5", "abstain_gate", "abstain" if abstain else "answer", detail)
    if abstain:
        job.abstained = True
        job.needs_review = True
        job.status = JobStatus.ABSTAINED
    elif job.needs_review or job.low_confidence:
        job.status = JobStatus.NEEDS_REVIEW
    else:
        job.status = JobStatus.COMPLETED
    return job


__all__ = ["tool_ingest", "tool_segment", "tool_recognize", "tool_translate", "tool_finalize"]
