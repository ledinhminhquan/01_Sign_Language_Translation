"""Neural gloss->text translator - a fine-tuned t5/m2m100 seq2seq (Colab; the Sign2Gloss2Text stage).

Wraps a HF seq2seq model that maps a space-joined gloss sequence ("HELLO ME NAME") to spoken text.
Fine-tuned by training/train_sign2text.py on (gloss, text) pairs from the synthetic generator (and,
when available, real gloss-text corpora). Offline the lexicon dictionary translator stands in.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

_PREFIX = "translate gloss to text: "


class NeuralTranslator:
    version = "neural-translator"

    def __init__(self, model, tokenizer, cfg: AppConfig):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg

    def translate(self, glosses: List[str]) -> str:
        import torch
        src = _PREFIX + " ".join(glosses)
        enc = self.tok(src, return_tensors="pt", truncation=True, max_length=self.cfg.model.max_gloss_len)
        with torch.no_grad():
            out = self.model.generate(**enc, max_length=self.cfg.model.max_text_len,
                                      num_beams=self.cfg.decode.beam_size)
        return self.tok.decode(out[0], skip_special_tokens=True)


def load_translator(cfg: AppConfig) -> Optional[NeuralTranslator]:
    """Load a fine-tuned translator if one was saved next to the recognizer; else None (lexicon)."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        path = cfg.model.output_dir / "translator"
        if not path.exists():
            return None
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
        model.eval()
        return NeuralTranslator(model, tok, cfg)
    except Exception as exc:
        logger.info("load_translator skipped (%s)", exc)
        return None


__all__ = ["NeuralTranslator", "load_translator", "_PREFIX"]
