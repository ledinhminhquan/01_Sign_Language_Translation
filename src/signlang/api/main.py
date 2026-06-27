"""FastAPI service for the sign-language translation system.

Endpoints
---------
* ``GET  /healthz`` / ``GET /readyz`` / ``GET /version``
* ``POST /translate`` - {seed | frames} -> glosses + spoken text + per-sign confidence + abstain flag

Provide ``seed`` to translate a generated synthetic signed sentence (demo), or ``frames`` for a real
pose-keypoint sequence (T x keypoint_dim). Low-confidence / out-of-vocabulary signing is flagged and
may be abstained on.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..data import lexicon
from ..data.synth_pose import PoseSequence, make_sentence
from ..logging_utils import get_logger
from .dependencies import get_agent, get_config
from .schemas import HealthResponse, TranslateRequest, TranslateResponse

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)


def _resp(out: dict) -> TranslateResponse:
    return TranslateResponse(glosses=out["glosses"], gloss_confs=out["gloss_confs"], text=out["text"],
                             n_segments=out["n_segments"], mean_conf=out["mean_conf"],
                             low_confidence=out["low_confidence"], abstained=out["abstained"],
                             needs_review=out["needs_review"], status=out["status"],
                             recognizer=out["recognizer"])


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(status="ok", recognizer=getattr(agent.recognizer, "name", "?"),
                          keypoint_dim=cfg.pose.keypoint_dim, vocab_size=cfg.data.vocab_size,
                          version=__version__)


@app.get("/readyz")
def readyz() -> dict:
    get_agent()
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    agent = get_agent()
    return {"app": __version__, "recognizer": getattr(agent.recognizer, "version", "centroid"),
            "model_version": cfg.serving.model_version}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    agent = get_agent()
    if req.frames is not None:
        import numpy as np
        seq = PoseSequence(frames=np.asarray(req.frames, dtype="float32"), fps=cfg.pose.fps, spec=None)
    elif req.seed is not None:
        seq = make_sentence(int(req.seed), cfg)
    else:
        raise HTTPException(status_code=422, detail="provide a 'seed' or a 'frames' array")
    return _resp(agent.translate(seq))


@app.get("/vocabulary")
def vocabulary() -> dict:
    return {"glosses": lexicon.gloss_to_text(cfg.data.vocab_size)}


__all__ = ["app"]
