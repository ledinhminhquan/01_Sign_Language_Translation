"""Pose/keypoint front-end - turns input into a per-frame keypoint sequence.

* ``MediaPipeEngine`` - extracts hand + body landmarks from a VIDEO (Colab; pretrained, NOT trained).
* ``SeedPoseEngine`` - the OFFLINE engine: the input is already a synthetic ``PoseSequence`` whose
  frames + embedded gold spec are returned as-is (no mediapipe/torch/video needed).
* ``StubPoseEngine`` - a trivial constant sequence (last-resort fallback).

``extract_pose`` returns a ``PoseSequence``; ``load_pose_engine`` picks the engine from config and
available libraries.
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import PoseConfig
from ..logging_utils import get_logger
from ..data.synth_pose import PoseSequence

logger = get_logger(__name__)


def _have(mod: str) -> bool:
    from importlib.util import find_spec
    try:
        return find_spec(mod) is not None
    except Exception:
        return False


class SeedPoseEngine:
    """Offline: passes a synthetic PoseSequence (frames + embedded gold) straight through."""

    name = "seed"

    def __init__(self, cfg: PoseConfig):
        self.cfg = cfg

    def extract(self, inp: Any) -> PoseSequence:
        if isinstance(inp, PoseSequence):
            seq = inp
        elif hasattr(inp, "frames"):
            seq = PoseSequence(frames=inp.frames, fps=getattr(inp, "fps", self.cfg.fps),
                               spec=getattr(inp, "spec", None))
        else:
            # a raw (T, D) array of pre-extracted keypoints, no gold spec
            seq = PoseSequence(frames=inp, fps=self.cfg.fps, spec=None)
        if self.cfg.normalize:
            seq = _normalize_seq(seq, self.cfg)
        return seq


class MediaPipeEngine:
    """Colab: extract hand + body landmarks from a video with MediaPipe Holistic."""

    name = "mediapipe"

    def __init__(self, cfg: PoseConfig):
        self.cfg = cfg

    def extract(self, inp: Any) -> PoseSequence:
        import numpy as np
        import mediapipe as mp  # lazy
        try:
            import cv2
        except Exception:
            cv2 = None
        holistic = mp.solutions.holistic.Holistic(static_image_mode=False)
        frames_kp = []
        for frame in _iter_video_frames(inp, cv2):
            res = holistic.process(frame)
            frames_kp.append(_landmarks_to_vec(res, self.cfg))
            if len(frames_kp) >= self.cfg.max_frames:
                break
        holistic.close()
        arr = np.asarray(frames_kp, dtype="float32") if frames_kp else \
            np.zeros((1, self.cfg.keypoint_dim), dtype="float32")
        seq = PoseSequence(frames=arr, fps=self.cfg.fps, spec=None)
        return _normalize_seq(seq, self.cfg) if self.cfg.normalize else seq


class StubPoseEngine:
    name = "stub"

    def __init__(self, cfg: PoseConfig):
        self.cfg = cfg

    def extract(self, inp: Any) -> PoseSequence:
        if isinstance(inp, PoseSequence):
            return inp
        import numpy as np
        return PoseSequence(frames=np.zeros((8, self.cfg.keypoint_dim), dtype="float32"),
                            fps=self.cfg.fps, spec=None)


def _normalize_seq(seq: PoseSequence, cfg: PoseConfig) -> PoseSequence:
    from .layout import normalize_sequence
    return PoseSequence(frames=normalize_sequence(seq.frames, cfg), fps=seq.fps, spec=seq.spec)


def _iter_video_frames(inp: Any, cv2):
    """Yield RGB frames from a path / numpy stack."""
    import numpy as np
    if isinstance(inp, str):
        if cv2 is None:
            raise RuntimeError("cv2 required to read a video path")
        cap = cv2.VideoCapture(inp)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
    else:
        for frame in np.asarray(inp):
            yield frame


def _landmarks_to_vec(res, cfg: PoseConfig):
    import numpy as np

    def block(landmarks, n):
        if landmarks is None:
            return np.zeros(n * cfg.coords, dtype="float32")
        pts = []
        for lm in list(landmarks.landmark)[:n]:
            pts += [lm.x, lm.y, getattr(lm, "z", 0.0)][: cfg.coords]
        v = np.asarray(pts, dtype="float32")
        if v.size < n * cfg.coords:
            v = np.concatenate([v, np.zeros(n * cfg.coords - v.size, dtype="float32")])
        return v

    return np.concatenate([block(res.left_hand_landmarks, cfg.n_hand_landmarks),
                           block(res.right_hand_landmarks, cfg.n_hand_landmarks),
                           block(res.pose_landmarks, cfg.n_body_landmarks)])


def load_pose_engine(cfg: PoseConfig, inp: Any = None):
    engine = cfg.engine
    if engine == "auto":
        if isinstance(inp, PoseSequence) or hasattr(inp, "spec"):
            engine = "seed"
        elif _have("mediapipe"):
            engine = "mediapipe"
        else:
            engine = "seed"
    if engine == "mediapipe" and _have("mediapipe"):
        return MediaPipeEngine(cfg)
    if engine == "stub":
        return StubPoseEngine(cfg)
    return SeedPoseEngine(cfg)


def extract_pose(inp: Any, cfg: PoseConfig, engine=None) -> PoseSequence:
    engine = engine or load_pose_engine(cfg, inp)
    return engine.extract(inp)


__all__ = ["SeedPoseEngine", "MediaPipeEngine", "StubPoseEngine", "load_pose_engine", "extract_pose"]
