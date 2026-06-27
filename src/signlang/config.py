"""Typed configuration + YAML loader for the signlang Sign Language Translation system.

Single source of truth for the pose/keypoint front-end (pretrained/algorithmic), the trainable
sign-to-gloss/text seq2seq core, the decoding + agent decision thresholds (D1-D5), the datasets,
and serving. Paths come from environment variables.

Pipeline (cascade): sign-language VIDEO -> pose/keypoint sequence (front-end, NOT trained) ->
segment into sign units -> RECOGNIZE glosses + TRANSLATE to spoken text (the trainable seq2seq
core) -> assemble. The only trained component is the sign-to-text seq2seq; the pose extractor is a
pretrained/algorithmic front-end (MediaPipe Holistic on Colab, a SeedEngine offline).

Why a synthetic spine: sign-language corpora are notoriously restrictively licensed and heavy
(video), so the PRIMARY offline data is a reproducible SYNTHETIC pose-sequence generator that
emits deterministic keypoint trajectories for a small sign vocabulary and embeds the gold
gloss/text; a SeedPoseEngine reads it back -> the whole pipeline runs with NO mediapipe/torch/video.

Environment overrides
---------------------
* ``SIGNLANG_ARTIFACTS_DIR`` - base for data/models/runs (Drive on Colab)
* ``SIGNLANG_DATA_DIR``      - dataset cache / generated synthetic pose sequences
* ``SIGNLANG_MODEL_DIR``     - trained models (the sign-to-text seq2seq)
* ``SIGNLANG_RUN_DIR``       - eval/benchmark/analysis JSON
* ``HF_HOME``               - HuggingFace cache
* ``SIGNLANG_LLM_API_KEY``   - optional key for the LLM agent brain
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("SIGNLANG_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("SIGNLANG_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("SIGNLANG_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def run_dir() -> Path:
    return Path(_env("SIGNLANG_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Synthetic pose-sequence generator (offline / tests) + optional real corpora.

    No permissively-licensed, lightweight "pose-sequence -> text" benchmark is reliably on the HF
    Hub, so the PRIMARY data is a reproducible SYNTHETIC generator
    (``data/synth_pose.py``): for each gloss in a small vocabulary emit a deterministic keypoint
    trajectory (hand + body landmarks over time), concatenate glosses into a sentence with a
    templated spoken-text translation, and embed the gold gloss/text. The SeedPoseEngine reads it
    back so recognition/translation/eval/agent run with no mediapipe/torch/video.

    Optional REAL datasets (VERIFY exact ids + licenses on the HF Hub before enabling; many
    sign-language corpora are research-only / non-commercial -> FLAG in docs/data_card.md).
    """
    # Real corpora (off by default; the synthetic generator is the spine). VERIFIED on the HF Hub:
    #   Sigurdur/icelandic-sign-language (Apache-2.0, tiny 214-row YouTube-SL-25 slice) = the ONLY cleanly
    #     permissive, viewer-loadable real corpus -> the real-data SMOKE TEST.
    #   PSewmuthu/How2Sign_Holistic (MIT tag, but derived from How2Sign NC -> FLAG) = MediaPipe Holistic
    #     pose/face/hand landmark sequences (the ideal pose-front-end input schema to mirror).
    #   Exploration-Lab/iSign (CC-BY-NC-SA, GATED -> FLAG) = defines SignPose2Text (this exact task).
    #   ALL continuous-SLT corpora are NC / gated / unspecified -> the synthetic generator is non-negotiable.
    real_smoke_dataset: str = "Sigurdur/icelandic-sign-language"   # Apache-2.0, permissive smoke test
    pose_dataset: str = "PSewmuthu/How2Sign_Holistic"              # MediaPipe landmarks (FLAG: NC upstream)
    gloss_text_dataset: str = "Exploration-Lab/iSign"             # SignPose2Text benchmark (FLAG: NC, gated)
    use_hf: bool = False                                          # default to the synthetic generator
    # Synthetic spine knobs:
    vocab_size: int = 40                  # distinct signs/glosses
    n_sentences: int = 600                # generated (pose-seq, gloss, text) training examples
    min_signs: int = 2                    # signs per sentence
    max_signs: int = 6
    frames_per_sign: int = 12             # keypoint frames emitted per sign
    seed: int = 42


@dataclass
class PoseConfig:
    """Pose/keypoint front-end (pretrained / algorithmic - NOT trained).

    On Colab, MediaPipe Holistic extracts hand + body landmarks per frame; offline the
    SeedPoseEngine reads the gold spec embedded in synthetic sequences. ``keypoint_dim`` is the
    per-frame feature size (here a compact hand+body landmark vector).
    """
    engine: str = "auto"                  # "auto"|"mediapipe"|"seed"|"stub"
    n_hand_landmarks: int = 21            # per hand (MediaPipe hand)
    n_body_landmarks: int = 25            # upper-body pose subset
    coords: int = 3                       # x, y, z (or visibility)
    fps: int = 25
    max_frames: int = 160                 # cap on a sequence
    normalize: bool = True                # center + scale-normalize keypoints

    @property
    def keypoint_dim(self) -> int:
        return (2 * self.n_hand_landmarks + self.n_body_landmarks) * self.coords


@dataclass
class ModelConfig:
    """The trainable sign-to-gloss/text seq2seq core.

    A compact transformer encoder-decoder over pose tokens (trained from scratch, since no
    permissively-licensed pretrained sign model is reliably available) maps the pose-keypoint
    sequence -> gloss tokens; an optional gloss->text stage reuses the P13/P14 MT pattern. The
    encoder projects each frame's keypoint vector into ``d_model`` and adds positional encodings.
    """
    arch: str = "pose2seq_transformer"
    d_model: int = 256
    n_heads: int = 4
    enc_layers: int = 4
    dec_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    # gloss -> text translation stage (reuse the MT backbone; small + permissive)
    text_backbone: str = "t5-small"               # Apache-2.0; m2m100_418M (MIT) for multilingual
    use_gloss_intermediate: bool = True           # Sign2Gloss2Text vs direct Sign2Text
    # training
    num_train_epochs: int = 20
    learning_rate: float = 3.0e-4
    per_device_train_batch_size: int = 64
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.1
    max_gloss_len: int = 24
    max_text_len: int = 48
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    seed: int = 42
    output_subdir: str = "sign2text"
    baseline_filename: str = "gloss_baseline.json"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir

    @property
    def baseline_path(self) -> Path:
        return self.output_dir / self.baseline_filename


@dataclass
class DecodeConfig:
    """Decoding for the seq2seq core."""
    beam_size: int = 1                    # greedy default (fast, deterministic); >1 = beam search
    max_decode_len: int = 48
    length_penalty: float = 1.0


@dataclass
class AgentConfig:
    """Sign-translation agent decision thresholds (D1-D5) + optional LLM brain."""
    # D1 - ingest / input gate
    min_frames: int = 6                   # too-short sequence -> fail
    # D2 - segmentation
    motion_quantile: float = 0.3          # rest/transition frames (low motion) split sign units
    min_segment_frames: int = 4
    # D3 - recognition confidence gate
    recog_min_conf: float = 0.15          # below -> mark the gloss low-confidence
    # D4 - translation verify
    verify_backtranslate: bool = True     # round-trip text->gloss agreement check (when available)
    min_chrf_keep: float = 0.20
    # D5 - abstain
    abstain_enabled: bool = True
    oov_abstain_ratio: float = 0.5        # > this fraction of segments OOV/low-conf -> abstain
    # optional cloud brain (off by default; the agent runs fully on rules)
    llm_fallback_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "SIGNLANG_LLM_API_KEY"


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Sign Language Translation API"
    api_version: str = "1.0.0"
    log_jobs: bool = True
    job_log_subdir: str = "job_logs"

    @property
    def job_log_path(self) -> Path:
        return run_dir() / self.job_log_subdir / "jobs.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Sign Language Translation System"
    author: str = "Le Dinh Minh Quan"
    student_id: str = "23127460"
    data: DataConfig = field(default_factory=DataConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "pose": PoseConfig, "model": ModelConfig, "decode": DecodeConfig,
             "agent": AgentConfig, "serving": ServingConfig}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author", "student_id") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(), "runs": run_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "PoseConfig", "ModelConfig", "DecodeConfig", "AgentConfig", "ServingConfig", "AppConfig",
           "load_config", "save_config", "ensure_dirs", "artifacts_dir", "data_dir", "model_dir", "run_dir"]
