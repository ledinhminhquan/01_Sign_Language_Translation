"""Job state + trace structures for the sign-translation agent (a deterministic FSM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class JobStatus(str, Enum):
    INGESTED = "ingested"
    SEGMENTED = "segmented"
    RECOGNIZED = "recognized"
    TRANSLATED = "translated"
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "summary": self.summary, "error": self.error}


@dataclass
class Decision:
    id: str                 # D1..D5
    name: str
    branch: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "branch": self.branch, "detail": self.detail}


@dataclass
class JobState:
    # input
    n_frames: int = 0
    source: str = "pose"               # "pose" | "video"
    # pipeline products
    segments: List[Tuple[int, int]] = field(default_factory=list)
    glosses: List[str] = field(default_factory=list)
    gloss_confs: List[float] = field(default_factory=list)
    text: str = ""
    # signals
    mean_conf: float = 0.0
    n_low_conf: int = 0
    low_confidence: bool = False
    abstained: bool = False
    needs_review: bool = False
    # bookkeeping
    status: JobStatus = JobStatus.INGESTED
    decisions: List[Decision] = field(default_factory=list)
    trace: List[ToolTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)

    def add_decision(self, id: str, name: str, branch: str, detail: str = "") -> None:
        self.decisions.append(Decision(id=id, name=name, branch=branch, detail=detail))

    def add_trace(self, trace: ToolTrace) -> None:
        self.trace.append(trace)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_frames": self.n_frames, "source": self.source, "segments": self.segments,
                "glosses": self.glosses, "gloss_confs": [round(c, 4) for c in self.gloss_confs],
                "text": self.text, "mean_conf": round(self.mean_conf, 4), "n_low_conf": self.n_low_conf,
                "low_confidence": self.low_confidence, "abstained": self.abstained,
                "needs_review": self.needs_review, "status": self.status.value,
                "decisions": [d.to_dict() for d in self.decisions],
                "trace": [t.to_dict() for t in self.trace], "metrics": self.metrics,
                "model_versions": self.model_versions}


__all__ = ["JobStatus", "ToolTrace", "Decision", "JobState"]
