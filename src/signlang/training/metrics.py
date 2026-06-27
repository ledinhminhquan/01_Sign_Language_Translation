"""Evaluation metrics - translation (BLEU/chrF/WER) + sign recognition (gloss WER/accuracy).

Pure-python (no sacrebleu/jiwer needed) so the whole eval runs with only numpy. BLEU is corpus
BLEU-4 with add-1 smoothing; chrF is the character n-gram F-score; WER is token edit distance.
Gloss recognition reuses the same edit-distance + a position-aligned accuracy.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple


# ── edit distance / WER ─────────────────────────────────────────────────────────

def _edit_distance(ref: Sequence, hyp: Sequence) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(list(ref_tokens), list(hyp_tokens)) / len(ref_tokens)


def corpus_wer(refs: List[str], hyps: List[str]) -> float:
    tot_err = tot_len = 0
    for r, h in zip(refs, hyps):
        rt, ht = r.split(), h.split()
        tot_err += _edit_distance(rt, ht)
        tot_len += max(1, len(rt))
    return round(tot_err / max(1, tot_len), 4)


# ── chrF ────────────────────────────────────────────────────────────────────────

def _char_ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else Counter()


def chrf(ref: str, hyp: str, max_n: int = 6, beta: float = 2.0) -> float:
    if not ref and not hyp:
        return 100.0
    precs, recs = [], []
    for n in range(1, max_n + 1):
        rg, hg = _char_ngrams(ref, n), _char_ngrams(hyp, n)
        nref, nhyp = sum(rg.values()), sum(hg.values())
        if nref == 0 and nhyp == 0:
            continue                       # skip orders too long for these strings (standard chrF)
        overlap = sum((rg & hg).values())
        precs.append(overlap / nhyp if nhyp else 0.0)
        recs.append(overlap / nref if nref else 0.0)
    if not precs:
        return 0.0
    p = sum(precs) / len(precs)
    r = sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return round(100.0 * (1 + b2) * p * r / (b2 * p + r), 4)


def corpus_chrf(refs: List[str], hyps: List[str]) -> float:
    if not refs:
        return 0.0
    return round(sum(chrf(r, h) for r, h in zip(refs, hyps)) / len(refs), 4)


# ── BLEU ──────────────────────────────────────────────────────────────────────

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(refs: List[str], hyps: List[str], max_n: int = 4) -> float:
    clipped = [0] * max_n
    totals = [0] * max_n
    ref_len = hyp_len = 0
    for r, h in zip(refs, hyps):
        rt, ht = r.split(), h.split()
        ref_len += len(rt)
        hyp_len += len(ht)
        for n in range(1, max_n + 1):
            rg, hg = _ngrams(rt, n), _ngrams(ht, n)
            tot = max(0, len(ht) - n + 1)
            totals[n - 1] += tot
            clipped[n - 1] += sum(min(c, rg[g]) for g, c in hg.items())
    precs = []
    for n in range(max_n):
        num = clipped[n] + 1.0          # add-1 smoothing
        den = totals[n] + 1.0
        precs.append(num / den)
    if min(precs) <= 0:
        return 0.0
    log_avg = sum(math.log(p) for p in precs) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
    return round(100.0 * bp * math.exp(log_avg), 4)


# ── aggregate ───────────────────────────────────────────────────────────────────

def translation_metrics(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    return {"bleu": corpus_bleu(refs, hyps), "chrf": corpus_chrf(refs, hyps),
            "wer": corpus_wer(refs, hyps), "n": len(refs)}


def recognition_metrics(ref_glosses: List[List[str]], hyp_glosses: List[List[str]]) -> Dict[str, float]:
    """Gloss WER (sequence edit distance) + position-aligned gloss accuracy + sequence exact match."""
    tot_err = tot_len = 0
    correct = aligned = exact = 0
    for ref, hyp in zip(ref_glosses, hyp_glosses):
        tot_err += _edit_distance(ref, hyp)
        tot_len += max(1, len(ref))
        for a, b in zip(ref, hyp):
            aligned += 1
            correct += int(a == b)
        exact += int(ref == hyp)
    n = max(1, len(ref_glosses))
    return {"gloss_wer": round(tot_err / max(1, tot_len), 4),
            "gloss_accuracy": round(correct / max(1, aligned), 4),
            "sequence_exact_match": round(exact / n, 4), "n": len(ref_glosses)}


__all__ = ["wer", "corpus_wer", "chrf", "corpus_chrf", "corpus_bleu",
           "translation_metrics", "recognition_metrics"]
