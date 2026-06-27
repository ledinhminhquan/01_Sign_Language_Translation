"""The sign-translation agent - a deterministic FSM over the cascade.

    ingest (D1) -> segment (D2) -> recognize (D3) -> translate+verify (D4) -> finalize/abstain (D5)

Holds the pose front-end + the recognizer + (optional) the neural translator. Runs fully offline
(SeedPoseEngine + numpy centroid recognizer + lexicon translator) and upgrades to MediaPipe + a
trained transformer recognizer + a t5 translator on Colab. Every step is timed and traced; it
abstains when too much of the utterance is out-of-vocabulary / low-confidence.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from ..config import AppConfig, ensure_dirs
from ..logging_utils import JsonlLogger, get_logger
from ..pose.engine import extract_pose, load_pose_engine
from . import tools
from .llm_orchestrator import LLMBrain
from .state import JobState, JobStatus, ToolTrace

logger = get_logger(__name__)


class TranslationAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, recognizer=None, translator=None,
                 load_model: bool = True):
        self.cfg = cfg or AppConfig()
        self.translator = translator
        if recognizer is None:
            from ..models.sign2text import load_recognizer
            recognizer = load_recognizer(self.cfg, prefer="neural" if load_model else "centroid")
        self.recognizer = recognizer
        if translator is None and load_model:
            try:
                from ..models.neural_translator import load_translator  # optional torch module
                self.translator = load_translator(self.cfg)
            except Exception as exc:
                logger.info("neural translator unavailable (%s); lexicon translator", exc)
        self.brain = LLMBrain(self.cfg.agent)
        ensure_dirs()
        self._log = JsonlLogger(self.cfg.serving.job_log_path) if self.cfg.serving.log_jobs else None

    def _step(self, job: JobState, name: str, fn: Callable[[], JobState], summary: str = "") -> JobState:
        t0 = time.perf_counter()
        try:
            job = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        job.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                summary=summary or name, error=err))
        return job

    def run(self, inp: Any, *, save: bool = True) -> JobState:
        job = JobState()
        t0 = time.perf_counter()
        engine = load_pose_engine(self.cfg.pose, inp)
        try:
            seq = extract_pose(inp, self.cfg.pose, engine=engine)
        except Exception as exc:
            logger.warning("pose extraction failed: %s", exc)
            job.status = JobStatus.FAILED
            return job
        job.model_versions["pose_engine"] = getattr(engine, "name", "?")

        job = self._step(job, "ingest", lambda: tools.tool_ingest(job, seq, self.cfg), "input gate (D1)")
        if job.status is not JobStatus.FAILED:
            job = self._step(job, "segment", lambda: tools.tool_segment(job, seq, self.cfg),
                             "motion segmentation (D2)")
            job = self._step(job, "recognize", lambda: tools.tool_recognize(job, seq, self.cfg, self.recognizer),
                             "gloss recognition + confidence (D3)")
            job = self._step(job, "translate", lambda: tools.tool_translate(job, self.cfg, self.translator),
                             "gloss->text + verify (D4)")
            job = self._step(job, "finalize", lambda: tools.tool_finalize(job, self.cfg), "abstain gate (D5)")

        if self.brain.available() and (job.abstained or job.low_confidence):
            note = self.brain.note(job.glosses, job.text, job.abstained)
            if note:
                job.metrics["brain_note"] = note
                job.metrics["brain_used"] = True

        job.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if save and self._log is not None:
            try:
                self._log.log("sign_translate", n_frames=job.n_frames, n_segments=len(job.segments),
                              glosses=job.glosses, text=job.text, mean_conf=round(job.mean_conf, 4),
                              low_confidence=job.low_confidence, abstained=job.abstained,
                              status=job.status.value, metrics=job.metrics)
            except Exception:
                pass
        return job

    def translate(self, inp: Any) -> dict:
        job = self.run(inp, save=False)
        return {"glosses": job.glosses, "gloss_confs": [round(c, 4) for c in job.gloss_confs],
                "text": job.text, "n_segments": len(job.segments), "mean_conf": round(job.mean_conf, 4),
                "low_confidence": job.low_confidence, "abstained": job.abstained,
                "needs_review": job.needs_review, "status": job.status.value,
                "recognizer": job.model_versions.get("recognizer", "?")}


_AGENT: Optional[TranslationAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> TranslationAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = TranslationAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["TranslationAgent", "get_agent"]
