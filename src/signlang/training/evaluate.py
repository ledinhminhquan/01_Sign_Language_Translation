"""Evaluation: sign recognition (gloss WER/accuracy) + translation (BLEU/chrF/WER) vs baselines.

Builds a held-out synthetic split (optionally with injected POSE NOISE), runs the agent end-to-end
(offline: SeedPose + numpy centroid recognizer + lexicon translator; Colab: MediaPipe + trained
transformer + t5), and scores:
  * recognition: gloss WER, position-aligned accuracy, sequence exact-match;
  * translation: BLEU-4 (headline), chrF, WER;
  * segmentation: boundary-F1 vs the gold sign boundaries;
  * baselines: most-frequent gloss, random gloss, identity-translate (gloss tokens as text),
    and the SEED ORACLE (perfect recognition -> upper bound on the translate stage);
  * abstention rate.
Results -> ``run_dir/eval.json``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..data import samples
from ..data.synth_pose import PoseSequence
from ..logging_utils import get_logger
from ..models import baseline as B
from ..segmentation.segmenter import boundary_f1, segment
from . import metrics as M

logger = get_logger(__name__)


def _add_pose_noise(seq: PoseSequence, sigma: float, seed: int) -> PoseSequence:
    if sigma <= 0:
        return seq
    import numpy as np
    rng = np.random.RandomState((seed * 9301 + 49297) & 0x7FFFFFFF)
    frames = np.asarray(seq.frames, dtype="float32") + rng.normal(scale=sigma, size=seq.frames.shape).astype("float32")
    return PoseSequence(frames=frames, fps=seq.fps, spec=seq.spec)


def evaluate(cfg: AppConfig, *, limit: Optional[int] = None, load_model: bool = True,
             pose_noise: float = 0.0, save: bool = True) -> Dict[str, Any]:
    from ..agent.translate_agent import TranslationAgent

    n = limit or 100
    ev = samples.seed_eval(cfg, n=n)
    if pose_noise > 0:
        ev = [{**ex, "seq": _add_pose_noise(ex["seq"], pose_noise, i)} for i, ex in enumerate(ev)]

    agent = TranslationAgent(cfg, load_model=load_model)

    ref_g, hyp_g, ref_t, hyp_t = [], [], [], []
    seg_f1, n_abstain = [], 0
    for ex in ev:
        out = agent.translate(ex["seq"])
        ref_g.append(ex["glosses"]); hyp_g.append(out["glosses"])
        ref_t.append(ex["text"]); hyp_t.append(out["text"])
        seg_f1.append(boundary_f1(segment(ex["seq"], cfg.agent), ex["boundaries"])["f1"])
        if out["abstained"]:
            n_abstain += 1

    recog = M.recognition_metrics(ref_g, hyp_g)
    trans = M.translation_metrics(ref_t, hyp_t)

    # baselines
    mf = B.build_most_frequent(cfg)
    rnd = B.build_random(cfg)
    mf_g = [[g for g, _ in mf.recognize(ex["seq"], ex["boundaries"])] for ex in ev]
    rnd_g = [[g for g, _ in rnd.recognize(ex["seq"], ex["boundaries"])] for ex in ev]
    identity_t = [B.identity_translate(ex["glosses"]) for ex in ev]   # oracle recog, no lexicon translate

    systems = {
        "agent": {"recognition": recog, "translation": trans},
        "most_frequent_gloss": {"recognition": M.recognition_metrics(ref_g, mf_g)},
        "random_gloss": {"recognition": M.recognition_metrics(ref_g, rnd_g)},
        "identity_translate": {"translation": M.translation_metrics(ref_t, identity_t)},
    }
    report = {
        "n": len(ev), "pose_noise": pose_noise,
        "systems": systems,
        "segmentation_boundary_f1": round(sum(seg_f1) / max(1, len(seg_f1)), 4),
        "abstain_rate": round(n_abstain / max(1, len(ev)), 4),
        "recognizer": getattr(agent.recognizer, "name", "?"),
        "model_version": getattr(agent.recognizer, "version", "centroid"),
        "headline": {
            "gloss_accuracy": recog["gloss_accuracy"], "gloss_wer": recog["gloss_wer"],
            "bleu": trans["bleu"], "chrf": trans["chrf"],
            "most_frequent_gloss_accuracy": systems["most_frequent_gloss"]["recognition"]["gloss_accuracy"],
            "identity_translate_bleu": systems["identity_translate"]["translation"]["bleu"],
            "segmentation_f1": round(sum(seg_f1) / max(1, len(seg_f1)), 4),
        },
    }
    if save:
        out = run_dir() / "eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("eval -> %s (gloss_acc=%s BLEU=%s seg_f1=%s noise=%s)", out,
                    recog["gloss_accuracy"], trans["bleu"], report["segmentation_boundary_f1"], pose_noise)
    return report


__all__ = ["evaluate"]
