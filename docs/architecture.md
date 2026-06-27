# P01 Sign Language Translation — System Architecture

Author: Le Dinh Minh Quan (student 23127460) · Package `signlang` · Folder `01_Sign_Language_Translation`

This document describes the end-to-end system architecture of P01: the **Sign2Gloss2Text cascade**, the module map under `src/signlang`, the data-flow between stages, and the offline / degradation design that lets the whole pipeline run with nothing but the Python standard library plus NumPy — no MediaPipe, no torch, no video, no network.

P01 is the last and hardest project of the assignment. Unlike the prior 18 text/OCR systems, the input here is a **video or a pose-keypoint sequence** and the output is a spoken-language **text sequence** — a sequence-to-sequence problem in a continuous spatio-temporal signal, not over characters or tokens.

---

## 1. What the system does

P01 maps a sign-language **video → spoken-language text**, via an intermediate **gloss** sequence (the now-universal Camgöz framing). A *gloss* is a written label for a single sign (`THANK-YOU`, `ME`, `BOOK`); a sentence is an ordered run of signs. Translation is non-trivial precisely because glosses are **not** the spoken sentence: glosses follow sign-language grammar and morphology, so `ME BOOK WANT` must become `i want a book` — re-ordering, function-word insertion, and a gloss→word lexicon all have to be learned.

The system is a **cascade**, not an end-to-end model:

> **pose front-end → segment → recognize gloss (the only trained stage) → translate gloss→text → assemble.**

Only the **sign-to-gloss/text core** is trained. The pose front-end is **pretrained / algorithmic and frozen** — MediaPipe Holistic is a Google computer-vision library, not a model we fit. This split is deliberate: it isolates a single small, measurable trainable unit (student-scale, ~5M params), and it mirrors the cascade discipline of P13 (s2st) / P14 (doctrans) / P15 (image-MT).

Wrapping the cascade is a **deterministic finite-state agent** (5 decision points) that gates on each stage's own intermediate signals — segment count, per-sign confidence, translation round-trip agreement — and decides whether to answer or to **abstain** with `needs_review`. That self-checking abstention ladder, not a bigger model, is the agentic value-add: a blind end-to-end decoder hallucinates fluent text from noise; this system refuses to.

---

## 2. High-level pipeline

Five logical stages, each tagged **PRETRAINED**, **ALGORITHMIC**, or **TRAINED**.

| # | Stage | Status | Component |
|---|-------|--------|-----------|
| 1 | Pose extraction | **PRETRAINED / ALGORITHMIC (frozen)** | MediaPipe Holistic — per-frame hand+body(+face) landmarks (Apache-2.0, Google). Offline: `SeedPoseEngine` reads gold embedded in the synthetic sequence. Permissive frozen-video alternative `microsoft/xclip-base-patch32` (MIT). |
| 2 | Motion segmentation | ALGORITHMIC | Velocity-based segmenter: near-still **rest frames** (low inter-frame landmark velocity) split the stream into sign units. No training. |
| 3 | Gloss recognition (CSLR) | **TRAINED** | Per-segment pose→gloss recognizer. Offline: a pure-NumPy nearest-centroid classifier over the segment's mean-pose displacement (no torch). On Colab: a compact transformer encoder over pose frames. **The trainable core, part 1.** |
| 4 | Gloss→text translation | **TRAINED** | Seq2seq translator `google-t5/t5-small` (Apache, 60.5M, default). Offline: the deterministic lexicon translator. **The trainable core, part 2.** |
| 5 | Assemble + verify | ALGORITHMIC | Concatenate per-sign text, attach per-sign confidence, run the round-trip / chrF keep-gate, emit glosses + text + abstain flag. |

Only stages 3–4 are fit; stages 1, 2, 5 are pretrained or pure geometry. There is **no pretrained Sign→Text or Gloss→Text checkpoint that loads cleanly into `transformers`** (verified on the Hub), which is exactly why the measured component is *our own* small seq2seq trained on a permissive backbone — see `model_selection.md`.

---

## 3. Data-flow diagram

```
                                   ┌──────────────────────────────────────────────┐
   video / pose-seq / seed-spec ──▶│  AGENT FSM  (src/signlang/agent)             │
                                   │  deterministic router, 5 decision points     │
                                   └──────────────────────────────────────────────┘
                                                     │
        ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
        ▼                                             ▼                                             ▼
 ┌─────────────┐  D1 frame-count gate     ┌────────────────────┐  route: video→pose vs already-pose
 │   INGEST    │── video → extract ──────▶│   POSE FRONT-END   │── video: MediaPipe Holistic ──────┐
 │ (frames ≥   │── pose .npy → pass ─────▶│  pose/engine       │   hand+body landmarks per frame    │
 │  min_frames)│── seed-spec → SeedPose ─▶│  + SeedPoseEngine  │── seed: read embedded gold pose ──│
 └─────────────┘                          └────────────────────┘                                    │
                                                     │  pose-keypoint sequence (T × D)              │
                                                     ▼   2×21 hand + 25 body × 3 coords             │
                                          ┌────────────────────┐                                    │
                                          │     SEGMENT        │  D2 motion-based segmentation       │
                                          │ segmentation/      │  low-velocity rest frames → split   │
                                          │   segmenter        │  → n sign segments / single span    │
                                          └────────────────────┘                                    │
                                                     │  segments: [(start, end, frames), …]         │
                                                     ▼                                              │
                                          ┌────────────────────┐  D3 per-segment gloss + confidence  │
                                          │     RECOGNIZE      │  numpy centroid (offline) OR        │
                                          │ models/neural_     │  transformer encoder (Colab)        │
                                          │   recognizer       │  conf < recog_min_conf → low-conf   │
                                          └────────────────────┘                                    │
                                                     │  gloss sequence + per-sign confidence ◀───────┘
                                                     ▼
                                          ┌────────────────────┐  D4 translate + verify
                                          │     TRANSLATE      │  lexicon (offline) OR t5-small      │
                                          │ models/neural_     │  round-trip text→gloss agreement /  │
                                          │   translator       │  chrF keep-gate → keep / re-flag    │
                                          └────────────────────┘
                                                     │  spoken text + verify flags
                                                     ▼
                                          ┌────────────────────┐  D5 finalize
                                          │     ASSEMBLE       │  low-conf ratio > oov_abstain_ratio │
                                          │  (sign2text)       │     → ABSTAIN ("uncertain",         │
                                          │                    │        needs_review)                │
                                          │                    │  else → glosses + text + per-sign   │
                                          └────────────────────┘        confidence
                                                     │
                                                     ▼
                                glosses + spoken text + per-sign confidence + abstain flag
                                          (+ segment boundaries, metrics, decision trace)
```

The five decision points (D1–D5) live in the agent, but each consults the stage it gates — the agent never re-implements pose extraction, segmentation, recognition, or translation; it only routes and decides how to present (or withhold) the result.

---

## 4. Module map (`src/signlang`)

The package mirrors the proven layout of P13 / P14 / P15; the pose front-end, the pose-sequence synthetic generator, the motion segmenter, the per-segment gloss recognizer, and the sign-translation agent are net-new for P01.

### `config`
Single source of truth for paths, the gloss vocabulary size, thresholds, and model ids. Holds the decision-point constants (`min_frames`, the rest-frame velocity threshold for segmentation, `recog_min_conf = 0.15`, the D4 chrF keep-gate, `oov_abstain_ratio = 0.5`), the model registry defaults (`google-t5/t5-small`), and the `SIGNLANG_OFFLINE` flag. Typed dataclass config reused from P13/P14 so every module reads one object.

### `pose/layout`
The **keypoint schema** — net-new and foundational for P01. Defines the per-frame landmark vector: **2×21 hand landmarks + 25 body (pose) landmarks × 3 coordinates** (x, y, z), the same shape MediaPipe Holistic and `PSewmuthu/How2Sign_Holistic` produce. Provides named index slices (left hand, right hand, body), the flat-vector dimensionality `D`, and helpers to assemble / slice a `(T × D)` sequence. Every other module speaks this layout, so swapping `SeedPoseEngine` for real MediaPipe output is transparent downstream.

### `pose/engine`
The **pose front-end** — frozen, never trained. On Colab, MediaPipe Holistic (Apache-2.0) reads a video and emits per-frame hand+body(+face) landmarks in the `pose/layout` schema; a permissive frozen-video alternative is `microsoft/xclip-base-patch32` (MIT) if going video→features instead of landmarks. Offline, the **`SeedPoseEngine`** reads the gold pose embedded in a synthetic sequence (no video decoding, no MediaPipe). A capability probe (`try import mediapipe` / file-type check) selects real-vs-seed at runtime on the **same code path**. License note: MediaPipe is an algorithmic Google library, so the front-end carries no checkpoint-license encumbrance; the **non-commercial video encoders `MCG-NJU/videomae-base` (CC-BY-NC) and `sign/mediapipe-vq` (CC-BY-NC-SA) are flagged and never shipped.**

### `data/lexicon`
The **gloss vocabulary + gloss→text map** — 40 ASL-style glosses, each with a deterministic motion direction seed (consumed by the generator) and a spoken-text expansion deliberately **≠** the gloss token (`THANK-YOU`→"thank you", `ME`→"i", `BOOK`→"a book"). This non-identity map is what makes the translate stage measurable: a passthrough baseline that copies gloss tokens cannot recover the lexicon or the word order.

### `data/synth_pose`
The **synthetic pose-sequence generator** — the **primary offline data source** and the defining design choice of P01, because **every continuous sign-language corpus on the Hub is non-commercial, gated, or unspecified** (see `data_card.md`). For each gloss the generator fixes a deterministic motion direction in keypoint space (seeded by gloss index); a sign is a **triangle stroke** (up-down) along that direction, so the per-sign **mean-pose displacement uniquely recovers the gloss**. Signs are separated by near-still **rest frames**, so the motion segmenter splits them. A sentence concatenates 2–6 signs; the spoken text comes from the lexicon. The gold spec `{glosses, text, boundaries}` is **embedded on the sequence**, so the `SeedPoseEngine` / `SeedRecognizer` read it back and the whole pipeline runs with no MediaPipe / torch / video / network.
- **Determinism:** per sample `i`, an RNG seeded from a base seed and `i`; sign trajectories and rest-frame counts are reproducible, so committed fixtures and exact metric scoring hold.
- **Robustness knobs:** controllable Gaussian pose noise (the robustness sweep), CLEAN mode (noise off, the recognition upper bound), and a pure-noise mode that must trigger abstention.

### `data/samples`
A handful of tiny committed fixtures — fixed-seed pose sequences with their gold `{glosses, text, boundaries}` — that drive unit tests and the smoke run without invoking the full generator.

### `data/dataset` + `data/download_dataset`
Loaders for the real corpora used as **smoke tests only** (the trained core is measured on the held-out synthetic split). The one cleanly permissive real corpus is **`Sigurdur/icelandic-sign-language`** (Apache-2.0, 214 rows, a YouTube-SL-25 slice; `video_id` + timed transcript) — the real-data smoke test; **`om192006/sign_language_keypoints`** (MIT, 29 isolated gestures) is the pose-schema template. `download_dataset` fetches these on demand and **flags every restrictive license** at load time: `PSewmuthu/How2Sign_Holistic` and `aipieces/How2Sign` (How2Sign upstream is CC-BY-NC), `Exploration-Lab/iSign` (CC-BY-NC-SA + **gated**; it defines *SignPose2Text* = this exact task), `Voxel51/WLASL` (license `other`), `Kibalama/poseformer-sign-language` (WLASL-derived). **RWTH-PHOENIX-2014T, CSL-Daily, and YouTube-ASL/SL-25 are not redistributable Hub repos — the loader does not invent ids for them.**

### `models/sign2text`
The **orchestrator** of the trainable core: given a pose sequence it runs segment → recognize → translate → assemble and returns `{glosses, text, per_sign_confidence, boundaries}`. It is the single object the agent and the API call; it composes the recognizer and translator below behind one interface so the offline (centroid + lexicon) and online (transformer + t5) cores are swapped without touching callers.

### `models/baseline`
The baselines that **isolate the trainable core** (see `translation_evaluation.md`): **most-frequent gloss** and **random gloss** recognizers; **identity / passthrough** translation (gloss tokens emitted as text — exposes the re-ordering and lexicon the model must learn); and the **Seed oracle** (perfect recognition → an upper bound on the translate stage). The trained core must beat the first three and approach the oracle.

### `models/neural_recognizer`
The per-segment **pose→gloss recognizer** — the trainable core, part 1. Offline it is a **pure-NumPy nearest-centroid classifier**: it computes each segment's mean-pose displacement and assigns the nearest gloss centroid — a genuine classifier of the sign's motion, **no torch**. On Colab it upgrades to a **compact transformer encoder** over the segment's pose frames (CTC-style), kept student-scale (~5M params). Architecture reference: `manohonsy/how2sign-pose-cslr` (MIT, 4.8M, pose+CTC CSLR on How2Sign) — proof a ~5M pose model is the right size. A capability probe selects centroid-vs-transformer; both return `(gloss, confidence)` per segment.

### `models/neural_translator`
The **gloss→text seq2seq translator** — the trainable core, part 2, ported from the P13/P14 MT harness. Default **`google-t5/t5-small`** (Apache-2.0, 60.5M, fits a T4). Offline it is the deterministic **lexicon translator** (`data/lexicon` map + simple gloss-grammar→spoken-order rules). Documented alternates: **`facebook/m2m100_418M`** (MIT, reuses the P13/P14 MT harness directly) and **`google/byt5-small`** (Apache, byte-level — robust to OOV glosses / symbolic pose tokens). Precedent for the "treat the recognized symbolic sequence as a source language and run standard MT" approach is `sign/sockeye-signwriting-to-text` (MIT, Sockeye — not HF transformers). The capability probe swaps the lexicon for the fine-tuned t5 when `transformers` + `torch` are present.

### `models/model_registry`
Maps logical roles (`recognizer`, `translator`, `pose_engine`) to concrete ids + their licenses, records the offline-stub fallback per role, and surfaces a license flag on every restrictive id. Template reused from P13/P14.

### `segmentation/segmenter`
The **motion-based sign segmenter** — net-new for P01, pure geometry, no training. Computes per-frame landmark **velocity** (frame-to-frame displacement in the `pose/layout` space); runs of low-velocity **rest frames** mark boundaries between signs. Emits `[(start, end), …]` segment spans (drives **D2**) and the boundary list scored against gold by boundary-F1. Robust to the rest-frame structure the generator builds in and to MediaPipe's real rest pauses alike.

### `training`
The seq2seq train/eval harness reused from P13/P14: fine-tunes the t5/m2m100 translator with the BLEU/chrF/WER metric set, and fits the transformer recognizer on the synthetic split. Corpus loading via `data`; checkpoints registered through `model_registry`. Trains exactly the two-part core — nothing else in the cascade is fit.

### `agent`
**Net-new for P01** — the mandatory agentic component: a deterministic FSM in `src/signlang/agent/` with **5 decision points** over states `ingest → segment → recognize → translate+verify → finalize`:
- **D1 ingest** — frame-count gate (`frames ≥ min_frames`) + route video→pose vs already-pose vs seed-spec; too-short / unreadable → `needs_review`.
- **D2 segment** — motion-based segmentation: low-velocity rest frames split signs into *n* segments; a degenerate stream collapses to a single span.
- **D3 recognize** — per-segment gloss + confidence; any segment below `recog_min_conf = 0.15` is flagged **low-confidence** (never silently translated).
- **D4 translate + verify** — gloss→text, then an optional **round-trip** text→gloss agreement check / chrF keep-gate; failures re-flag the sign rather than emitting unverified text.
- **D5 finalize** — **abstain** if the low-confidence / OOV segment ratio exceeds `oov_abstain_ratio = 0.5` → returns `"uncertain"` + `needs_review`; otherwise returns glosses + spoken text + per-sign confidence.

An optional LLM brain (`anthropic`) is **OFF by default, advisory only** (e.g. a "please repeat the sign" note) and **never changes the output**. The agent runs fully offline on `SeedPoseEngine` + NumPy centroid + lexicon. Value-add: sign segmentation + per-sign confidence gating + translation verification + **abstention** — strictly more than a blind end-to-end decode. See `agent_architecture.md`.

### `api`
**FastAPI** service: `POST /translate` accepts either a `seed` spec or raw `frames` (pose sequence) and returns glosses + spoken text + per-sign confidence + the abstain flag. A **Gradio** demo UI is mounted for interactive use. Packaged with **Docker** (mediapipe + ffmpeg + libGL) and an HF Space. The offline backend-model-free path (SeedPoseEngine + NumPy centroid + lexicon) serves on CPU.

### `analysis`
Computes and tabulates the metric suite from agent runs (see `translation_evaluation.md`): recognition — **gloss-WER** (sub/del/ins) + position-aligned gloss accuracy + sequence exact-match; translation — **BLEU-1..4 (BLEU-4 = headline)** + chrF + WER (reused from P13/P14); segmentation — **boundary-F1** vs gold sign boundaries; plus the **abstention rate**. Verified offline seed numbers: clean synthetic gloss accuracy **1.0**, BLEU **99+**, segmentation-F1 **1.0** vs a most-frequent-gloss floor of **0.02** and identity-translate BLEU **~84** (the lexicon adds ~15 BLEU); a pose-noise sweep degrades recognition gracefully and pure-noise input **abstains**. The module also surfaces the **honesty caveat** — automatic SLT metrics (BLEU/chrF/ROUGE/BLEURT) are length-sensitive and blind to hallucination / semantic equivalence (Yazdani et al. 2025, hf.co/papers/2510.25434) — in both the tables and the autoreport.

### `autoreport`
Auto-generates the run report (config + metric tables + decision-trace counts + sample gloss/text outputs + the metric-reliability caveat) from a single command. Template reused from P13/P14.

### `monitoring`
Run-time signal capture: per-stage timings, decision-point branch counts (how many runs answered vs abstained, how many segments were low-confidence), and recognition-confidence / segment-count distributions. Template reused from P13/P14; feeds `continual_learning_monitoring.md`.

### `automation`
Autopilot driver that chains generate → pose → segment → recognize → translate → evaluate → report end-to-end for reproducible benchmark runs. Reused template.

### `grading`
Self-grading harness that scores a run against the project rubric (metrics present, baselines beaten, offline path green, all 5 agent decisions exercised, abstention fires on noise) for the assignment deliverable. Reused template.

---

## 5. Offline & degradation design

The defining engineering property of P01 is that the **entire cascade runs deterministically with only the Python standard library plus NumPy** — no MediaPipe binary, no torch, no transformers, no video files, no network. Tests pass in CI/Colab with nothing downloaded. Four mechanisms make this work.

### 5.1 Lazy imports + capability probes (one code path)
Heavy dependencies (`mediapipe`, `torch`, `transformers`, `cv2`/video decoding, `sacrebleu`, `anthropic`) are imported lazily **inside** the functions that need them, never at module top level. Each stage runs a capability probe — `try import` / file-type check — and selects the real component when present, the stub when absent. The env flag `SIGNLANG_OFFLINE=1` pins stub mode for reproducible tests. Crucially, **the probe upgrades each stage in place; the surrounding code and the tests are identical online and offline.**

### 5.2 SeedPoseEngine (offline pose front-end)
`SeedPoseEngine` is the offline replacement for MediaPipe. For a synthetic sequence it returns the gold pose embedded in the spec — in the exact `pose/layout` schema (2×21 hand + 25 body × 3) — without decoding any video, so segmentation + recognition + translation + metrics execute end-to-end offline. When `mediapipe` and a real video are present, the probe switches to MediaPipe Holistic on the **same downstream code** — every module past the front-end is schema-agnostic to which engine produced the frames.

### 5.3 NumPy centroid recognizer + lexicon translator + pure-Python metrics
- **NumPy nearest-centroid recognizer** (no torch): per segment it computes the mean-pose displacement and assigns the nearest gloss centroid — a *genuine* motion classifier (it actually discriminates the synthetic strokes, not a lookup), reused as **both** the offline recognizer **and** a baseline reference. Swaps to the transformer encoder when torch is present.
- **Lexicon translator** (no transformers): the `data/lexicon` gloss→text map plus simple gloss-order→spoken-order rules — deterministic, instant, no download, and the identity/passthrough baseline as a foil. Swaps to fine-tuned t5-small when transformers+torch are present.
- **Metrics** (no sacrebleu/jiwer): gloss-WER and translation-WER use a pure-Python two-row Levenshtein; BLEU and chrF use minimal pure-Python implementations, with real sacrebleu when installed (a tolerance test asserts they agree). Boundary-F1 and accuracy are pure set/sequence arithmetic — already dependency-free and fully meaningful offline.

### 5.4 Sequence-level pose normalization
Before recognition, every pose sequence is normalized in the `pose/layout` space — **per-sequence** centering and scaling (e.g. translate to a body-relative origin, scale by torso/shoulder span) so that signer position, distance from the camera, and frame size do not leak into the displacement features the recognizer reads. This is the same normalization for synthetic and real MediaPipe input, which is what lets the offline centroid recognizer's geometry transfer to the on-Colab transformer, and which underpins the signer-independence robustness claim (a model keyed to absolute pixel coordinates would fail across signers and cameras).

**Net:** with stdlib + NumPy only, `seed-spec → SeedPoseEngine → segment → recognize(centroid) → translate(lexicon) → all metrics → agent → abstention` runs deterministically. The offline numbers are *meaningful*, not just wiring checks — gloss accuracy, boundary-F1, and WER measure genuine deterministic geometry, and abstention provably fires on pure-noise input — while the trained Colab core is what raises real-data quality.

---

## 6. Reuse map

- **From P13 (s2st) / P14 (doctrans):** the BLEU / chrF / WER metric implementations and the seq2seq train/eval pattern (→ the gloss→text translator and the translation metric suite); the t5 / m2m100 backbone; the config / logging / model-registry / autoreport / monitoring / automation / grading / CLI / API templates.
- **From P15 / P17 / P19 / P20:** the **embedded-gold synthetic-generator** + Seed/Stub offline pattern (→ `data/synth_pose` + `SeedPoseEngine` + `SeedRecognizer`), and the capability-probe one-code-path discipline.
- **New for P01:** the **pose-keypoint layout** (`pose/layout`) and front-end (`pose/engine`, MediaPipe / SeedPoseEngine); the synthetic **pose-sequence** generator (motion *trajectories*, not images); the motion-based **segmenter** (`segmentation/segmenter`); the per-segment **gloss recognizer** (NumPy nearest-centroid offline + transformer on Colab); sequence-level pose normalization; and the 5-decision **sign-translation agent**.

---

## 7. Ethics, privacy & robustness (architectural commitments)

Sign-language video is **biometric and identifying** — it captures faces and hands, and it is **Deaf-community data**. The architecture therefore: processes locally / on-device by default, **retains no raw video** by default (the pose schema and the abstain-aware agent need only landmarks, not pixels), requires consent for any retention, and keeps the optional LLM brain **off**. **Representation bias is acute**: a recognizer trained on one sign language or one signer set fails on others — so the system **never presents a low-confidence translation as authoritative**. The D3 per-sign confidence gate, the D4 translation verification, the per-sign confidence surfaced in the API response, and the **D5 abstention** are the architectural expression of "keep a human in the loop," especially in medical and legal settings. SLT here is an **assistive aid, not a replacement for human interpreters**, and the project commits to engaging the Deaf community. Robustness is engineered, not assumed: noisy pose (low-quality video) is exercised by the generator's noise sweep, continuous-vs-isolated signing and multi-sign **segmentation** by the rest-frame segmenter and boundary-F1, signer-independence by sequence-level normalization, and **out-of-vocabulary signs** by the low-confidence → abstain path — a designed degradation, not a bug. The restrictive-licensing reality of the field is met head-on by the synthetic spine: a reproducible, permissive primary data source that owes nothing to a non-commercial corpus. See `privacy_robustness.md` and `ethics_statement.md`.
