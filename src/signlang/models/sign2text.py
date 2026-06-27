"""The sign-to-gloss/text core: per-segment gloss recognition + gloss->text translation.

Recognizers (pose segment -> gloss + confidence):
* ``CentroidRecognizer`` - the OFFLINE trainable core: a nearest-centroid classifier fit on
  (segment-feature, gloss) pairs. The segment feature is the sign's mean-pose displacement
  (mean frame over the span minus the sequence mean) - distinct per gloss, so this genuinely
  classifies with NO torch. This is the measured model offline.
* ``NeuralRecognizer`` - the Colab upgrade: a transformer encoder over the segment frames trained
  with cross-entropy (see training/train_sign2text.py); loaded if a trained model + torch exist.
* ``SeedRecognizer`` - an oracle that reads the embedded gold spec (upper bound / last resort).

Translation (gloss sequence -> spoken text): the lexicon dictionary offline; a t5/m2m100 backbone
is the optional fine-tuned upgrade (Sign2Gloss2Text).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig, ModelConfig
from ..logging_utils import get_logger
from ..data import lexicon
from ..data.synth_pose import PoseSequence

logger = get_logger(__name__)


# ── segment feature ────────────────────────────────────────────────────────────

def segment_feature(frames, span: Tuple[int, int]):
    """Mean-pose displacement of a sign span (numpy vector, L2-normalized)."""
    import numpy as np
    a = np.asarray(frames, dtype="float32")
    s, e = span
    seg = a[s:e] if e > s else a
    if seg.shape[0] == 0:
        return np.zeros(a.shape[1], dtype="float32")
    feat = seg.mean(axis=0) - a.mean(axis=0)
    n = float(np.linalg.norm(feat)) + 1e-6
    return (feat / n).astype("float32")


# ── recognizers ────────────────────────────────────────────────────────────────

class SeedRecognizer:
    """Reads the embedded gold glosses (oracle); aligns to the requested spans by order."""

    name = "seed"
    version = "seed"

    def recognize(self, seq: PoseSequence, spans: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
        spec = getattr(seq, "spec", None) or {}
        glosses = list(spec.get("glosses", []))
        out: List[Tuple[str, float]] = []
        for i in range(len(spans)):
            out.append((glosses[i], 1.0) if i < len(glosses) else ("", 0.0))
        return out


class CentroidRecognizer:
    """Nearest-centroid gloss classifier over segment features (numpy; the offline trainable core)."""

    name = "centroid"

    def __init__(self, vocab: List[str], centroids=None, version: str = "centroid-1.0"):
        self.vocab = vocab
        self.centroids = centroids        # np.ndarray (V, D) normalized
        self.version = version

    def fit(self, feats, glosses: List[str]) -> "CentroidRecognizer":
        import numpy as np
        D = feats.shape[1] if hasattr(feats, "shape") else len(feats[0])
        sums = {g: np.zeros(D, dtype="float32") for g in self.vocab}
        counts = {g: 0 for g in self.vocab}
        for f, g in zip(feats, glosses):
            if g in sums:
                sums[g] += np.asarray(f, dtype="float32")
                counts[g] += 1
        cents = []
        for g in self.vocab:
            c = sums[g] / counts[g] if counts[g] else np.zeros(D, dtype="float32")
            n = float(np.linalg.norm(c)) + 1e-6
            cents.append(c / n)
        self.centroids = np.stack(cents, axis=0)
        return self

    def _scores(self, feat):
        import numpy as np
        return self.centroids @ np.asarray(feat, dtype="float32")   # cosine (both normalized)

    def recognize(self, seq: PoseSequence, spans: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
        import numpy as np
        out: List[Tuple[str, float]] = []
        for span in spans:
            feat = segment_feature(seq.frames, span)
            scores = self._scores(feat)
            # softmax confidence over the top scores
            z = scores - scores.max()
            p = np.exp(z * 4.0)
            p = p / p.sum()
            idx = int(np.argmax(p))
            out.append((self.vocab[idx], float(p[idx])))
        return out

    def save(self, path: str | Path) -> Path:
        import numpy as np
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p / "centroids.npz", centroids=self.centroids)
        (p / "vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Optional["CentroidRecognizer"]:
        import numpy as np
        p = Path(path)
        if not (p / "centroids.npz").exists():
            return None
        cents = np.load(p / "centroids.npz")["centroids"]
        vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        return cls(vocab=vocab, centroids=cents)


def fit_centroid_recognizer(cfg: AppConfig, n_train: Optional[int] = None) -> CentroidRecognizer:
    """Fit the offline recognizer on segments of the synthetic training collection.

    The frames are passed through the SAME pose front-end (normalization) the agent uses at
    inference, so train + inference features match exactly.
    """
    import numpy as np
    from ..data.synth_pose import make_sentence
    from ..pose.engine import extract_pose
    vocab = lexicon.vocab(cfg.data.vocab_size)
    feats, glosses = [], []
    n = n_train or min(300, cfg.data.n_sentences)
    for i in range(n):
        raw = make_sentence(cfg.data.seed + i, cfg)
        seq = extract_pose(raw, cfg.pose)                  # normalize identically to the agent path
        for span, g in zip(raw.spec["boundaries"], raw.spec["glosses"]):
            feats.append(segment_feature(seq.frames, span))
            glosses.append(g)
    feats = np.stack(feats, axis=0) if feats else np.zeros((1, cfg.pose.keypoint_dim), dtype="float32")
    return CentroidRecognizer(vocab=vocab).fit(feats, glosses)


def load_recognizer(cfg: AppConfig, seq: Optional[PoseSequence] = None, prefer: str = "neural"):
    """Pick a recognizer: trained neural > fitted centroid > seed oracle."""
    if prefer == "neural":
        try:
            from .model_registry import resolve_latest
            from .neural_recognizer import NeuralRecognizer  # optional torch module
            latest = resolve_latest(cfg.model.output_dir)
            if latest is not None:
                rec = NeuralRecognizer.load(latest, cfg)
                if rec is not None:
                    return rec
        except Exception as exc:
            logger.info("neural recognizer unavailable (%s); using centroid", exc)
    try:
        return fit_centroid_recognizer(cfg)
    except Exception as exc:
        logger.info("centroid fit failed (%s); using seed oracle", exc)
        return SeedRecognizer()


# ── translation (gloss sequence -> spoken text) ─────────────────────────────────

def translate(glosses: List[str], cfg: AppConfig, model=None) -> str:
    """Gloss sequence -> spoken text. Offline = the lexicon dictionary; t5/m2m100 is the upgrade."""
    if model is not None:
        try:
            return model.translate(glosses)
        except Exception as exc:
            logger.info("neural translator failed (%s); lexicon fallback", exc)
    return lexicon.translate_glosses(glosses, cfg.data.vocab_size)


__all__ = ["segment_feature", "SeedRecognizer", "CentroidRecognizer", "fit_centroid_recognizer",
           "load_recognizer", "translate"]
