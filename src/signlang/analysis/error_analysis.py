"""Error analysis (offline): exact / abstained / wrong buckets + worst examples.

Runs the agent on the held-out synthetic split and buckets each sentence: exact (text matches gold),
abstained, or wrong. Short keys ``correct``/``abstained``/``wrong`` feed the charts.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def error_analysis(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
    cfg = cfg or AppConfig()
    try:
        from ..agent.translate_agent import TranslationAgent
        from ..data import samples
        agent = TranslationAgent(cfg, load_model=False)
        ev = samples.seed_eval(cfg, n=limit or 80)
    except Exception as exc:
        return _stub(str(exc), save)

    correct = abstained = wrong = 0
    worst: List[Dict] = []
    for ex in ev:
        try:
            out = agent.translate(ex["seq"])
        except Exception:
            continue
        if out["abstained"]:
            abstained += 1
            continue
        if out["text"].strip() == ex["text"].strip():
            correct += 1
        else:
            wrong += 1
            if len(worst) < 8:
                worst.append({"gold": ex["text"], "pred": out["text"],
                              "gold_glosses": ex["glosses"], "pred_glosses": out["glosses"]})
    n = max(1, len(ev))
    result = {"n": len(ev), "correct": correct, "abstained": abstained, "wrong": wrong,
              "exact_text_rate": round(correct / n, 4), "abstain_rate": round(abstained / n, 4),
              "worst_examples": worst}
    if save:
        _save(result)
    logger.info("error analysis: correct=%d abstained=%d wrong=%d", correct, abstained, wrong)
    return result


def _stub(error: str, save: bool) -> Dict:
    result = {"n": 0, "correct": 0, "abstained": 0, "wrong": 0, "exact_text_rate": 0.0,
              "abstain_rate": 0.0, "worst_examples": [], "error": error}
    if save:
        _save(result)
    return result


def _save(result: Dict) -> None:
    try:
        d = run_dir() / "error_analysis"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"errors-{utc_stamp()}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.info("error_analysis: could not save (%s)", exc)


__all__ = ["error_analysis"]
