"""Neural per-segment gloss recognizer - a compact transformer encoder over pose frames (Colab).

The trainable core (the measured component on GPU): each sign segment's frame sequence is projected
to ``d_model``, encoded by a small transformer, mean-pooled, and classified into the gloss
vocabulary. Trained with cross-entropy (see training/train_sign2text.py). Imported only when torch
is available; offline the numpy CentroidRecognizer stands in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import AppConfig
from ..logging_utils import get_logger
from ..data.synth_pose import PoseSequence

logger = get_logger(__name__)


def build_module(cfg: AppConfig, n_classes: int):
    import torch
    import torch.nn as nn

    mc = cfg.model
    kp = cfg.pose.keypoint_dim

    class PoseEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(kp, mc.d_model)
            self.pos = nn.Parameter(torch.zeros(1, cfg.pose.max_frames, mc.d_model))
            layer = nn.TransformerEncoderLayer(d_model=mc.d_model, nhead=mc.n_heads,
                                               dim_feedforward=mc.ff_dim, dropout=mc.dropout,
                                               batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=mc.enc_layers)
            self.head = nn.Linear(mc.d_model, n_classes)

        def forward(self, x, mask=None):                  # x: (B, T, kp)
            h = self.proj(x) + self.pos[:, : x.shape[1], :]
            h = self.enc(h, src_key_padding_mask=mask)
            if mask is not None:
                keep = (~mask).unsqueeze(-1).float()
                pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            else:
                pooled = h.mean(1)
            return self.head(pooled)

    return PoseEncoder()


class NeuralRecognizer:
    name = "neural"

    def __init__(self, module, vocab: List[str], cfg: AppConfig, version: str = "neural"):
        self.module = module
        self.vocab = vocab
        self.cfg = cfg
        self.version = version
        self._torch = __import__("torch")

    def _segment_tensor(self, seq: PoseSequence, span: Tuple[int, int]):
        import numpy as np
        torch = self._torch
        s, e = span
        seg = np.asarray(seq.frames, dtype="float32")[s:e]
        if seg.shape[0] == 0:
            seg = np.zeros((1, self.cfg.pose.keypoint_dim), dtype="float32")
        cap = self.cfg.pose.max_frames
        if seg.shape[0] > cap:
            seg = seg[:cap]
        return torch.tensor(seg).unsqueeze(0)

    def recognize(self, seq: PoseSequence, spans: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
        torch = self._torch
        self.module.eval()
        out: List[Tuple[str, float]] = []
        with torch.no_grad():
            for span in spans:
                x = self._segment_tensor(seq, span)
                logits = self.module(x)
                p = torch.softmax(logits, dim=-1)[0]
                idx = int(torch.argmax(p))
                out.append((self.vocab[idx], float(p[idx])))
        return out

    def save(self, path: str | Path) -> Path:
        torch = self._torch
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), p / "recognizer.pt")
        (p / "vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path, cfg: AppConfig) -> Optional["NeuralRecognizer"]:
        try:
            import torch
            p = Path(path)
            if not (p / "recognizer.pt").exists():
                return None
            vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
            module = build_module(cfg, len(vocab))
            module.load_state_dict(torch.load(p / "recognizer.pt", map_location="cpu"))
            return cls(module, vocab, cfg, version=p.name)
        except Exception as exc:
            logger.info("NeuralRecognizer.load failed (%s)", exc)
            return None


__all__ = ["build_module", "NeuralRecognizer"]
