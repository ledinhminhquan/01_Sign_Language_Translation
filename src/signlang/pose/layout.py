"""Keypoint layout for the pose front-end.

A compact hand + upper-body landmark layout (a subset of MediaPipe Holistic): two hands of
``n_hand`` landmarks + ``n_body`` upper-body landmarks, each with ``coords`` channels (x, y, z).
The per-frame feature vector is their flattened concatenation. Kept tiny + deterministic so the
synthetic generator and a real MediaPipe extraction share the same dimensionality.
"""

from __future__ import annotations

from typing import List, Tuple

from ..config import PoseConfig


def keypoint_dim(cfg: PoseConfig) -> int:
    return cfg.keypoint_dim


def segments(cfg: PoseConfig) -> List[Tuple[str, int]]:
    """(name, n_landmarks) blocks that make up a frame vector, in order."""
    return [("left_hand", cfg.n_hand_landmarks), ("right_hand", cfg.n_hand_landmarks),
            ("body", cfg.n_body_landmarks)]


def normalize_sequence(frames, cfg: PoseConfig):
    """Sequence-level normalize a (T, D) keypoint array.

    Each frame is CENTERED on its own keypoint centroid (removes signer position / translation),
    but the whole sequence is scaled by a SINGLE factor (the global keypoint RMS) - so per-frame
    *velocities* are only uniformly rescaled, never distorted. (Per-frame independent scaling would
    break motion-based segmentation.) numpy in/out.
    """
    import numpy as np
    a = np.asarray(frames, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] == 0:
        return a
    T = a.shape[0]
    pts = a.reshape(T, -1, cfg.coords)                      # (T, K, C)
    pts = pts - pts.mean(axis=1, keepdims=True)             # center each frame on its keypoints
    scale = float(np.sqrt((pts ** 2).sum(axis=2).mean()) + 1e-6)   # one global scale
    pts = pts / scale
    return pts.reshape(T, -1).astype("float32")


__all__ = ["keypoint_dim", "segments", "normalize_sequence"]
