"""Train the sign-to-gloss/text core (the trainable component) - Colab/GPU.

Two stages:
  1. POSE -> GLOSS recognizer: a transformer encoder over each sign segment's frames, trained with
     cross-entropy on (segment, gloss) pairs from the synthetic generator (gold boundaries).
  2. (optional) GLOSS -> TEXT translator: fine-tune ``t5-small`` on (gloss-sequence, text) pairs.

bf16/tf32 on Ampere+/H100. Skipped offline (no torch); the numpy centroid recognizer + lexicon
translator stand in. On real data, swap the synthetic loader for a pose-corpus loader.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config import AppConfig
from ..data import lexicon
from ..data.synth_pose import make_sentence
from ..logging_utils import get_logger
from ..models import model_registry as reg
from ..pose.engine import extract_pose

logger = get_logger(__name__)


def _segment_examples(cfg: AppConfig, n: int) -> Tuple[List, List[int], List[str]]:
    """(segment_frames, gloss_idx, gloss) from synthetic sentences using the gold boundaries."""
    vocab = lexicon.vocab(cfg.data.vocab_size)
    v2i = {g: i for i, g in enumerate(vocab)}
    segs, labels, glosses = [], [], []
    for i in range(n):
        raw = make_sentence(cfg.data.seed + i, cfg)
        seq = extract_pose(raw, cfg.pose)
        for span, g in zip(raw.spec["boundaries"], raw.spec["glosses"]):
            s, e = span
            segs.append(seq.frames[s:e])
            labels.append(v2i[g])
            glosses.append(g)
    return segs, labels, glosses


def train_recognizer(cfg: AppConfig, limit: Optional[int] = None) -> Dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from ..models.neural_recognizer import NeuralRecognizer, build_module

    mc = cfg.model
    torch.manual_seed(mc.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(mc.tf32)
    vocab = lexicon.vocab(cfg.data.vocab_size)
    n = limit or min(cfg.data.n_sentences, 800)
    segs, labels, _ = _segment_examples(cfg, n)
    cap = cfg.pose.max_frames

    class SegDS(Dataset):
        def __len__(self):
            return len(segs)

        def __getitem__(self, i):
            x = np.asarray(segs[i], dtype="float32")[:cap]
            return x, labels[i]

    def collate(batch):
        xs, ys = zip(*batch)
        T = max(x.shape[0] for x in xs)
        B = len(xs)
        kp = cfg.pose.keypoint_dim
        out = np.zeros((B, T, kp), dtype="float32")
        mask = np.ones((B, T), dtype=bool)
        for b, x in enumerate(xs):
            out[b, : x.shape[0]] = x
            mask[b, : x.shape[0]] = False
        return torch.tensor(out), torch.tensor(mask), torch.tensor(ys, dtype=torch.long)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = build_module(cfg, len(vocab)).to(device)
    loader = DataLoader(SegDS(), batch_size=mc.per_device_train_batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(module.parameters(), lr=mc.learning_rate)
    lossf = nn.CrossEntropyLoss(label_smoothing=mc.label_smoothing)

    module.train()
    for epoch in range(mc.num_train_epochs):
        tot = 0.0
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            logits = module(x, mask=mask)
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
            tot += float(loss)
        logger.info("recognizer epoch %d/%d loss=%.4f", epoch + 1, mc.num_train_epochs, tot / max(1, len(loader)))

    module.cpu()
    version = reg.make_version("pose2gloss")
    final_dir = mc.output_dir / version
    NeuralRecognizer(module, vocab, cfg, version=version).save(final_dir)
    reg.write_metadata(final_dir, version=version, base_model=mc.arch,
                       dataset_signature={"segments": len(segs), "seed": cfg.data.seed},
                       metrics={}, extra={"type": "pose2gloss-transformer", "n_classes": len(vocab)})
    reg.update_latest_pointer(mc.output_dir, final_dir)
    logger.info("recognizer training done -> %s", final_dir)
    return {"version": version, "model_dir": str(final_dir), "n_segments": len(segs), "n_classes": len(vocab)}


def train_translator(cfg: AppConfig, limit: Optional[int] = None) -> Dict:
    """Fine-tune t5-small on (gloss-sequence, text) pairs (the Gloss->Text stage)."""
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from datasets import Dataset as HFDataset

    from ..models.neural_translator import _PREFIX

    mc = cfg.model
    n = limit or min(cfg.data.n_sentences, 800)
    src, tgt = [], []
    for i in range(n):
        s = make_sentence(cfg.data.seed + i, cfg)
        src.append(_PREFIX + " ".join(s.spec["glosses"]))
        tgt.append(s.spec["text"])
    tok = AutoTokenizer.from_pretrained(mc.text_backbone)
    model = AutoModelForSeq2SeqLM.from_pretrained(mc.text_backbone)

    def prep(batch):
        enc = tok(batch["src"], truncation=True, max_length=mc.max_gloss_len)
        lab = tok(text_target=batch["tgt"], truncation=True, max_length=mc.max_text_len)
        enc["labels"] = lab["input_ids"]
        return enc

    ds = HFDataset.from_dict({"src": src, "tgt": tgt}).map(prep, batched=True,
                                                           remove_columns=["src", "tgt"])
    out_dir = mc.output_dir / "translator"
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "_hf"), per_device_train_batch_size=mc.per_device_train_batch_size,
        learning_rate=5e-4, num_train_epochs=max(3, mc.num_train_epochs), logging_steps=20,
        save_strategy="no", report_to=[], bf16=bool(mc.bf16 and torch.cuda.is_available()))
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model))
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    logger.info("translator fine-tune done -> %s", out_dir)
    return {"translator_dir": str(out_dir), "n_pairs": len(src), "backbone": mc.text_backbone}


def train_all(cfg: AppConfig, limit: Optional[int] = None, train_translator_stage: bool = True) -> Dict:
    out = {"recognizer": train_recognizer(cfg, limit=limit)}
    if train_translator_stage and cfg.model.use_gloss_intermediate:
        try:
            out["translator"] = train_translator(cfg, limit=limit)
        except Exception as exc:
            logger.warning("translator stage skipped (%s)", exc)
            out["translator"] = {"skipped": str(exc)}
    return out


__all__ = ["train_recognizer", "train_translator", "train_all"]
