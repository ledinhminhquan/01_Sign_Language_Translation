"""Synthetic pose-sequence generator - the offline spine of the system.

For each gloss in the vocabulary we fix a deterministic *motion signature* (a unit direction in
keypoint space, seeded by the gloss index). A sign is rendered as a smooth up-down stroke along
that direction (so the per-sign mean displacement recovers the gloss - a classifier/seq2seq can
learn it), separated by near-still REST frames (so a motion-based segmenter can split signs). A
sentence concatenates 2-6 signs; its spoken text comes from the gloss->text lexicon. The gold spec
(glosses, text, segment boundaries) is embedded on the sequence so the SeedPoseEngine can read it
back - the whole pipeline runs with NO mediapipe / torch / video.

Mirrors the embedded-gold synthetic pattern of P15 (doc images) / P17 (scenes) / P19/P20.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig, DataConfig, PoseConfig
from ..logging_utils import get_logger
from . import lexicon

logger = get_logger(__name__)


@dataclass
class PoseSequence:
    """A pose-keypoint sequence (T frames x keypoint_dim) + the embedded gold spec."""
    frames: Any                                  # np.ndarray (T, D) float32
    fps: int = 25
    spec: Optional[Dict[str, Any]] = field(default=None)   # {glosses, text, boundaries}

    @property
    def n_frames(self) -> int:
        try:
            return int(self.frames.shape[0])
        except Exception:
            return len(self.frames)


def _rng(seed: int):
    import numpy as np
    return np.random.RandomState(seed & 0x7FFFFFFF)


def _gloss_direction(gloss_idx: int, dim: int):
    """A fixed unit motion direction for a gloss (deterministic by index)."""
    import numpy as np
    r = _rng(1000 + gloss_idx)
    u = r.normal(size=dim).astype("float32")
    n = float(np.linalg.norm(u)) + 1e-6
    return u / n


def _base_pose(dim: int):
    import numpy as np
    # a fixed neutral resting pose (small structured pattern, not all-zeros)
    idx = np.arange(dim, dtype="float32")
    return 0.1 * np.sin(idx * 0.3)


def _sign_frames(gloss_idx: int, n: int, dim: int, amp: float, noise: float, seed: int):
    import numpy as np
    base = _base_pose(dim)
    u = _gloss_direction(gloss_idx, dim)
    r = _rng(seed)
    out = np.zeros((n, dim), dtype="float32")
    # TRIANGLE envelope (0 -> peak -> 0): constant per-frame velocity throughout the stroke (no
    # zero-velocity midpoint), so a motion segmenter does not split one sign in two; the non-zero
    # MEAN displacement (~amp/2 along u) still encodes the gloss for the recognizer.
    for t in range(n):
        env = 1.0 - abs(2.0 * t / max(1, n - 1) - 1.0)
        out[t] = base + u * (amp * env) + r.normal(scale=noise, size=dim).astype("float32")
    return out


def _rest_frames(n: int, dim: int, noise: float, seed: int):
    import numpy as np
    base = _base_pose(dim)
    r = _rng(seed)
    return base[None, :] + r.normal(scale=noise * 0.4, size=(n, dim)).astype("float32")


def make_sentence(seed: int, cfg: Optional[AppConfig] = None) -> PoseSequence:
    cfg = cfg or AppConfig()
    dc: DataConfig = cfg.data
    pc: PoseConfig = cfg.pose
    dim = pc.keypoint_dim
    rng = random.Random(seed)
    vocab = lexicon.vocab(dc.vocab_size)

    n_signs = rng.randint(dc.min_signs, dc.max_signs)
    glosses = [rng.choice(vocab) for _ in range(n_signs)]
    fps = dc.frames_per_sign

    import numpy as np
    chunks: List[Any] = []
    boundaries: List[Tuple[int, int]] = []
    cursor = 0
    rest_len = max(2, fps // 3)
    # lead-in rest
    chunks.append(_rest_frames(rest_len, dim, 0.02, seed)); cursor += rest_len
    for i, g in enumerate(glosses):
        gi = vocab.index(g)
        sf = _sign_frames(gi, fps, dim, amp=1.0, noise=0.05, seed=seed * 31 + i)
        start = cursor
        chunks.append(sf); cursor += fps
        boundaries.append((start, cursor))
        chunks.append(_rest_frames(rest_len, dim, 0.02, seed * 7 + i)); cursor += rest_len

    frames = np.concatenate(chunks, axis=0).astype("float32")
    if pc.max_frames and frames.shape[0] > pc.max_frames:
        frames = frames[: pc.max_frames]
        boundaries = [(s, e) for (s, e) in boundaries if e <= pc.max_frames]
    text = lexicon.translate_glosses(glosses, dc.vocab_size)
    spec = {"glosses": glosses, "text": text, "boundaries": boundaries}
    return PoseSequence(frames=frames, fps=pc.fps, spec=spec)


def motion_profile(frames) -> List[float]:
    """Per-frame velocity magnitude (||frame_t - frame_{t-1}||); used by the segmenter."""
    import numpy as np
    a = np.asarray(frames, dtype="float32")
    if a.shape[0] < 2:
        return [0.0] * a.shape[0]
    vel = np.linalg.norm(np.diff(a, axis=0), axis=1)
    return [0.0] + [float(v) for v in vel]


def read_spec(seq: PoseSequence) -> Optional[Dict[str, Any]]:
    return getattr(seq, "spec", None)


def save_sequence(seq: PoseSequence, path: str | Path) -> Path:
    import numpy as np
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, frames=np.asarray(seq.frames, dtype="float32"))
    (p.with_suffix(".json")).write_text(json.dumps(seq.spec, ensure_ascii=False), encoding="utf-8")
    return p


def generate_collection(out_dir: str | Path, n: Optional[int] = None, cfg: Optional[AppConfig] = None,
                        seed: Optional[int] = None) -> Dict[str, Any]:
    cfg = cfg or AppConfig()
    n = n or cfg.data.n_sentences
    seed = cfg.data.seed if seed is None else seed
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for i in range(n):
        seq = make_sentence(seed + i, cfg)
        rel = f"seq_{i:05d}.npz"
        save_sequence(seq, out / rel)
        manifest.append({"id": f"seq_{i:05d}", "file": rel, "glosses": seq.spec["glosses"],
                         "text": seq.spec["text"], "n_frames": seq.n_frames})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("generated %d synthetic pose sequences -> %s", n, out)
    return {"out_dir": str(out), "n": n, "keypoint_dim": cfg.pose.keypoint_dim}


__all__ = ["PoseSequence", "make_sentence", "motion_profile", "read_spec", "save_sequence",
           "generate_collection"]
