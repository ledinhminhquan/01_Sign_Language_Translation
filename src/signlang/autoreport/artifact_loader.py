"""Collect generated run artifacts into one dict for the report + slides generators.

Reads the JSON under ``run_dir()`` - the eval (recognition + translation vs baselines + segmentation
+ abstain), the error analysis, the quality report, a latency benchmark, the pose-noise tune, and a
monitoring snapshot - plus the trained-model metadata. Every read is defensive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppConfig, run_dir
from ..models.model_registry import read_metadata, resolve_latest


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def load_artifacts(cfg: AppConfig) -> Dict[str, Any]:
    rd = run_dir()
    arts: Dict[str, Any] = {
        "eval": _load_json(rd / "eval.json"),
        "error_analysis": _load_json(rd / "error_analysis" / "latest.json"),
        "quality": _load_json(rd / "quality" / "latest.json"),
        "benchmark": _load_json(rd / "benchmark" / "latest.json"),
        "tune": _load_json(rd / "tune" / "tune.json"),
        "monitoring": _load_json(rd / "monitoring" / "latest.json"),
    }
    try:
        latest = resolve_latest(cfg.model.output_dir)
        arts["model_meta"] = read_metadata(latest) if latest else {}
    except Exception:
        arts["model_meta"] = {}
    return arts


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _systems(arts: Dict[str, Any]) -> Dict[str, Any]:
    return (arts.get("eval") or {}).get("systems") or {}


def agent_recog(arts: Dict[str, Any], key: str) -> Optional[float]:
    return _num(((_systems(arts).get("agent") or {}).get("recognition") or {}).get(key))


def agent_trans(arts: Dict[str, Any], key: str) -> Optional[float]:
    return _num(((_systems(arts).get("agent") or {}).get("translation") or {}).get(key))


def baseline_recog(arts: Dict[str, Any], system: str, key: str) -> Optional[float]:
    return _num(((_systems(arts).get(system) or {}).get("recognition") or {}).get(key))


def baseline_trans(arts: Dict[str, Any], system: str, key: str) -> Optional[float]:
    return _num(((_systems(arts).get(system) or {}).get("translation") or {}).get(key))


def headline(arts: Dict[str, Any], key: str) -> Optional[float]:
    return _num(((arts.get("eval") or {}).get("headline") or {}).get(key))


def seg_f1(arts: Dict[str, Any]) -> Optional[float]:
    return _num((arts.get("eval") or {}).get("segmentation_boundary_f1"))


def abstain_rate(arts: Dict[str, Any]) -> Optional[float]:
    return _num((arts.get("eval") or {}).get("abstain_rate"))


def has_eval(arts: Dict[str, Any]) -> bool:
    return bool(_systems(arts))


def recognizer_name(arts: Dict[str, Any]) -> str:
    return str((arts.get("eval") or {}).get("recognizer") or "centroid")


def model_version(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("version") or (arts.get("eval") or {}).get("model_version") or "untrained (centroid)")


def base_model(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("base_model") or "pose2seq_transformer + t5-small")


def buckets(arts: Dict[str, Any]) -> Dict[str, Optional[float]]:
    ea = arts.get("error_analysis") or {}
    return {"correct": _num(ea.get("correct")), "abstained": _num(ea.get("abstained")),
            "wrong": _num(ea.get("wrong"))}


def latency(arts: Dict[str, Any], pct: str = "p50") -> Optional[float]:
    b = (arts.get("benchmark") or {}).get("latency_ms") or {}
    return _num(b.get(pct))


def read_doc(name: str) -> str:
    p = repo_root() / "docs" / name
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


__all__ = ["load_artifacts", "read_doc", "repo_root", "agent_recog", "agent_trans", "baseline_recog",
           "baseline_trans", "headline", "seg_f1", "abstain_rate", "has_eval", "recognizer_name",
           "model_version", "base_model", "buckets", "latency"]
