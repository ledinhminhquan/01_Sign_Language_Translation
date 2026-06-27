"""Pydantic request/response schemas for the sign-translation API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    seed: Optional[int] = Field(None, description="Generate a synthetic signed sentence from this seed and translate it")
    frames: Optional[List[List[float]]] = Field(None, description="A pose-keypoint sequence (T x keypoint_dim)")


class TranslateResponse(BaseModel):
    glosses: List[str] = []
    gloss_confs: List[float] = []
    text: str = ""
    n_segments: int = 0
    mean_conf: float = 0.0
    low_confidence: bool = False
    abstained: bool = False
    needs_review: bool = False
    status: str = ""
    recognizer: str = ""
    disclaimer: str = ("Sign-language translation is assistive, NOT a substitute for a qualified human "
                       "interpreter. Per-sign confidence is shown; low-confidence / out-of-vocabulary signing "
                       "is flagged ('needs review') and may be abstained on. Pose video is biometric data.")


class HealthResponse(BaseModel):
    status: str
    recognizer: str
    keypoint_dim: int
    vocab_size: int
    version: str


__all__ = ["TranslateRequest", "TranslateResponse", "HealthResponse"]
