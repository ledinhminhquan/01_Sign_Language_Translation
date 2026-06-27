# P01 — Sign Language Translation · Project Plan

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment — the last and hardest project.
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`

This plan turns the locked design (see `docs/DESIGN_BRIEF.md`) into an executable schedule. P01 is the first
**video / pose-keypoint sequence → text** task in the assignment series (the prior 18 were text/OCR), so the plan
front-loads the genuinely new engineering — the pose-keypoint layout, the synthetic pose-sequence generator, the
motion-based segmenter, and the per-segment gloss recognizer — and reuses the MT, synthetic-spine, and infrastructure
patterns proven in P13/P14/P15/P17/P19/P20.

The defining constraint, confirmed during research, is that **there is no permissively-licensed, directly-loadable
Sign→Text/Gloss→Text seq2seq checkpoint and no permissive continuous-SLT corpus on the Hub** — every continuous
benchmark is non-commercial, gated, or unspecified. The plan is therefore built around a reproducible **synthetic
pose-sequence spine** as primary data, with one cleanly-licensed real corpus (`Sigurdur/icelandic-sign-language`,
Apache-2.0) reserved as a real-data smoke test. The schedule below is sequenced so the entire offline path
(generate → segment → recognize → translate → eval → agent → tests) runs with **no MediaPipe, no torch, no video,
no network** before any Colab/GPU work begins.

---

## 1. Objectives and definition of done

The cascade is **Sign2Gloss2Text**:

```
VIDEO ──(MediaPipe Holistic, frozen / algorithmic — NOT trained)──► pose-keypoint sequence
      ──► motion-based SEGMENT into sign units
      ──► RECOGNIZE gloss per segment  (the trainable core)
      ──► TRANSLATE gloss → spoken text  (the trainable core)
      ──► assemble sentence
```

Only the **sign-segment → gloss recognizer** and the **gloss → text translator** are trained. The pose front-end is
pretrained / algorithmic and frozen. "Done" means:

- **D1 — Offline spine works end-to-end on CPU.** `data/synth_pose.py` generates sentences (2–6 signs each) over a
  40-gloss lexicon; `SeedPoseEngine` and `SeedRecognizer` read the embedded gold back; the numpy nearest-centroid
  recognizer genuinely classifies pose displacement (no torch); the lexicon translator emits spoken text.
- **D2 — Metrics fire on a held-out synthetic split.** Recognition gloss-WER + position-aligned gloss accuracy +
  sequence exact-match; translation BLEU-1..4 (BLEU-4 headline) + chrF/chrF++ + ROUGE-L + WER; segmentation
  boundary-F1; abstention rate. All four baselines (identity-translate, most-frequent gloss, random gloss, Seed
  oracle) report, and the trained core beats 1–3 and approaches 4.
- **D3 — The agent's 5 decision points all fire** on at least one trace each (ingest gate, segment, recognize,
  translate+verify, finalize/abstain), with pure-noise input correctly abstaining.
- **D4 — Colab upgrade reproduces the headline numbers** with the real MediaPipe Holistic front-end + transformer
  recognizer + `t5-small` translator, plus a passing real-data smoke test on the Icelandic slice.
- **D5 — Deploy artifacts ship:** FastAPI `POST /translate`, Gradio demo, Docker image (mediapipe + ffmpeg + libGL),
  HF Space; offline backend-free path (SeedPose + centroid + lexicon) runs on CPU.
- **D6 — All 14 Section-I docs complete**, including the honesty caveat on SLT metric reliability (Yazdani et al.,
  2025, hf.co/papers/2510.25434) surfaced in `translation_evaluation`, the autoreport, and the model card.

---

## 2. Pipeline diagram (with offline vs Colab split)

```
                         ┌─────────────────── FRONT-END (frozen, NOT trained) ───────────────────┐
  raw input              │                                                                       │
  ─────────              │   OFFLINE:  SeedPoseEngine  ── reads gold embedded in synth sequence   │
   video.mp4   ──────────┤                                                                       │
   pose .npy   ──────────┤   COLAB:    MediaPipe Holistic ── 2×21 hand + 25 body landmarks ×3     │
                         └───────────────────────────────────┬───────────────────────────────────┘
                                                             ▼
                                          pose-keypoint sequence  (pose/layout.py)
                                                             │
                                   ┌─────────────────────────▼─────────────────────────┐
                                   │  D2  MOTION-BASED SEGMENTER                         │
                                   │  low-velocity REST frames split signs → boundaries │
                                   └─────────────────────────┬─────────────────────────┘
                                                             ▼
                                   ┌───────────────── TRAINABLE CORE ──────────────────┐
                                   │  D3  RECOGNIZER  per-segment pose → gloss + conf   │
                                   │      offline: numpy nearest-centroid (no torch)    │
                                   │      colab:   compact transformer encoder (~5M)    │
                                   │                                                    │
                                   │  D4  TRANSLATOR  gloss → spoken text               │
                                   │      offline: lexicon map                          │
                                   │      colab:   t5-small (default) / m2m100 / byt5   │
                                   │      + round-trip verify / chrF keep-gate          │
                                   └─────────────────────────┬─────────────────────────┘
                                                             ▼
                                   ┌─────────────────────────────────────────────────┐
                                   │  D5  FINALIZE / ABSTAIN                            │
                                   │  if low-conf segment ratio > oov_abstain_ratio    │
                                   │     → "uncertain" + needs_review                  │
                                   │  else → glosses + text + per-sign confidence      │
                                   └─────────────────────────┬─────────────────────────┘
                                                             ▼
                                              FastAPI / Gradio / autoreport
```

The agent (`src/signlang/agent/`) is the deterministic FSM wrapping this pipeline; D1 (the ingest frame-count gate
and video↔pose routing) precedes the segmenter. The optional LLM brain (`anthropic`) is OFF by default and advisory
only — it can attach a "please repeat" note but never changes the output.

---

## 3. Milestones and timeline

The plan is organized into **9 milestones over ~9 weeks** of part-time student effort. Weeks are indicative; the hard
dependency is that the offline spine (M2–M5) is fully green on CPU before any GPU spend (M6). Each milestone ends with
a concrete, testable deliverable.

| # | Milestone | Key deliverables | Depends on | Est. effort | Week |
|---|-----------|------------------|------------|-------------|------|
| **M0** | Research & licensing lock | `DESIGN_BRIEF.md`; verified model/dataset ids on the Hub; license flags for every NC/gated/unspecified source; reuse map from P13/P14/P15 | — | 3 d | 1 |
| **M1** | Keypoint layout & lexicon | `pose/layout.py` (2×21 hand + 25 body × 3 coords, mirroring MediaPipe / `PSewmuthu/How2Sign_Holistic`); `data/lexicon.py` (40 glosses + gloss→text map where text ≠ gloss) | M0 | 2 d | 1 |
| **M2** | Synthetic pose generator + SeedPoseEngine | `data/synth_pose.py` (per-gloss deterministic motion direction; triangle stroke; rest-frame separators; 2–6-sign sentences; embedded gold `{glosses, text, boundaries}`); `SeedPoseEngine` reads gold; held-out split | M1 | 4 d | 2 |
| **M3** | Motion-based segmenter | velocity profile → low-velocity rest detection → sign boundaries; boundary-F1 vs gold; tuning of rest threshold / min-segment length | M2 | 3 d | 2–3 |
| **M4** | Centroid recognizer (offline core) | pure-numpy nearest-centroid classifier on per-sign mean-pose displacement; per-segment confidence; `SeedRecognizer` oracle; gloss-WER + position-aligned accuracy + seq-exact-match | M2, M3 | 4 d | 3 |
| **M5** | Translator + metrics + baselines | lexicon translator; reuse P13/P14 BLEU/chrF/WER + ROUGE-L; baselines (identity-translate, most-frequent, random, Seed oracle); robustness sweep (pose-noise → recognition degradation); abstention on pure noise | M4 | 4 d | 4 |
| **M6** | Neural upgrade on Colab | MediaPipe Holistic extraction; compact transformer recognizer (~5M, ref `manohonsy/how2sign-pose-cslr`); `t5-small` gloss→text (reuse P13/P14 seq2seq train pattern); reproduce headline numbers; real-data smoke test on `Sigurdur/icelandic-sign-language` | M5 | 5 d | 5–6 |
| **M7** | Agent (5-decision FSM) | `src/signlang/agent/` ingest→segment→recognize→translate+verify→finalize; confidence gating; round-trip verify / chrF keep-gate; abstention; traced decisions; optional advisory `anthropic` brain (OFF default); all 5 decisions fire in tests | M5 (offline), M6 (neural) | 4 d | 6–7 |
| **M8** | Eval, autoreport, monitoring, grading | full metric run + charts; autoreport with the **SLT-metric reliability caveat**; monitoring/continual-learning hooks; grading harness; CLI | M5, M7 | 3 d | 7–8 |
| **M9** | Deploy & docs | FastAPI `POST /translate`; Gradio demo; Docker (mediapipe + ffmpeg + libGL); HF Space; all 14 Section-I docs incl. model card, ethics, privacy/robustness, slide deck | M6, M7, M8 | 4 d | 8–9 |

**Critical path:** M0 → M1 → M2 → M3/M4 → M5 → (M6 ∥ M7) → M8 → M9. M3 and M4 can overlap once the generator (M2)
emits both boundaries and per-sign motion. M6 (Colab/GPU) and M7 (agent, offline-first) can proceed in parallel after
M5 because the agent is built and tested against the offline spine before the neural core lands.

**Hard gate:** no GPU/Colab session is opened until M5 is green on CPU — this keeps GPU spend bounded and ensures the
neural work only has to reproduce, not discover, the headline numbers.

---

## 4. Risk register

| ID | Risk | Likelihood | Impact | Mitigation | Trigger / owner action |
|----|------|-----------|--------|------------|------------------------|
| **R1** | **Corpus licensing** — every continuous-SLT corpus is non-commercial (How2Sign CC-BY-NC, iSign CC-BY-NC-SA + gated), gated, or unspecified; none is a clean redistributable Hub repo | Certain | High | Synthetic pose-sequence generator is the **primary, non-negotiable** data; every restrictive id flagged in `data_card`; only `Sigurdur/icelandic-sign-language` (Apache-2.0) and `om192006/sign_language_keypoints` (MIT) used directly, and only as smoke tests | Already mitigated by design; never train the *measured* core on NC/gated data |
| **R2** | **No loadable Sign→Text checkpoint** — no pretrained SLT model loads cleanly into `transformers` | Certain | Medium | The trained core is **our own** small seq2seq (transformer recognizer + `t5-small`); pretrained ids (`manohonsy/how2sign-pose-cslr`, `sign/sockeye-signwriting-to-text`) used as architecture references only, not as the measured component | Already mitigated; do not depend on any external SLT checkpoint |
| **R3** | **Pose quality** — real MediaPipe landmarks from low-quality video are noisy / drop frames, unlike clean synthetic poses | Medium | High | Robustness sweep injects pose noise offline (M5) to characterize degradation before Colab; confidence gating + abstention catch low-quality input; smoke test on real Icelandic video exposes real noise | If smoke-test recognition collapses, widen rest-threshold tolerance and lower `recog_min_conf` review band |
| **R4** | **Signer-independence / representation bias** — a model trained on one signer set / one sign language fails on others | High | High | Never present low-confidence output as authoritative: abstain (D5) + per-sign confidence + human-in-the-loop; documented in `privacy_robustness` + `ethics_statement`; framed as assistive, not interpreter-replacing | Surface confidence in API/Gradio; flag for review in medical/legal contexts |
| **R5** | **Segmentation errors** — continuous signing has no clean rest frames; over-/under-segmentation cascades into gloss-WER | Medium | Medium | Motion-based segmenter with tunable rest threshold + min-segment length; boundary-F1 tracked as a first-class metric; gloss-WER's ins/del terms expose segmentation-induced errors; synthetic generator provides gold boundaries for direct tuning | If boundary-F1 drops on real data, add velocity smoothing / hysteresis |
| **R6** | **Metric unreliability** — BLEU/chrF/ROUGE/BLEURT are length-sensitive and blind to hallucination / semantic equivalence for SLT (Yazdani et al. 2025) | Certain | Medium | Report the **full** standard set AND flag the limitation prominently in `translation_evaluation`, the autoreport, and the model card; lean on the Seed-oracle upper bound and exact-match for ground truth on synthetic data | Always pair any BLEU claim with the caveat |
| **R7** | **Scope creep** — this is the hardest project; the new pose/segment/recognize stack could overrun | Medium | Medium | Offline-first; reuse P13/P14 MT + P15/P17/P19/P20 infra wholesale; hard CPU-green gate before GPU; numpy centroid keeps the offline core trivially debuggable | If M2–M5 slip, ship offline-only and treat Colab neural upgrade as stretch |
| **R8** | **Privacy** — sign-language video is biometric + identifying (face/hands, Deaf-community data) | Certain | High | On-device/edge processing path; no retention by default; LLM brain OFF; consent framing in `ethics_statement`; offline SeedPose path needs no video at all | Default-deny retention; document in `privacy_robustness` |
| **R9** | **GPU availability / cost** — Colab T4/L4 quotas, A100/H100 contention | Low | Low | Models are deliberately tiny (~5M recognizer, 60.5M `t5-small`) → fit a free/Pro T4; A100/H100 only if scaling the recognizer; offline path needs no GPU at all | Fall back to T4; HF Pro for Space hosting |

---

## 5. Resource needs

**Compute**

- **CPU only (primary, M0–M5, M7-offline, M8–M9 docs):** the entire offline spine — synthetic generation, segmenter,
  numpy nearest-centroid recognizer, lexicon translator, metrics, agent, tests, FastAPI/Gradio backend-free path —
  runs on a laptop CPU with no torch, no MediaPipe, no video, no network.
- **Colab T4 (16 GB) — default GPU (M6):** sufficient for `t5-small` (60.5M) gloss→text training and the compact
  ~5M transformer recognizer (`manohonsy/how2sign-pose-cslr` confirms a ~5M pose model is student-scale). Free or Pro.
- **Colab L4 / A100 (optional):** only if scaling the recognizer or moving to a frozen-video front-end
  (`microsoft/xclip-base-patch32`, MIT). Not on the critical path.
- **H100:** not required; flagged only as a notional ceiling for a larger byt5/m2m100 translator experiment.

**Services & accounts**

- **Hugging Face Pro** — for the HF Space (deploy demo) and faster Hub I/O; Hub access authenticated as
  `ledinhminhquan` (used to verify all model/dataset ids in M0).
- **Anthropic API key** — only for the optional advisory LLM brain, OFF by default; not needed for any measured result.

**Software / system deps**

- Offline: Python, numpy, the `signlang` package (config/logging/registry/autoreport/charts/CLI templates reused).
- Colab/Docker: `mediapipe`, `ffmpeg`, `libGL` (system libs for video decode + landmark extraction), `torch`,
  `transformers`, `sacrebleu`/`sacremoses` (BLEU/chrF reused from P13/P14).

**Data**

- Primary: the synthetic pose-sequence generator (ours, no license encumbrance).
- Smoke tests: `Sigurdur/icelandic-sign-language` (Apache-2.0, 214 rows), `om192006/sign_language_keypoints` (MIT).
- Reference only (flagged, **not** used to train the measured core): `PSewmuthu/How2Sign_Holistic`, `aipieces/How2Sign`,
  `Exploration-Lab/iSign`, `Kibalama/poseformer-sign-language`, `Voxel51/WLASL`.

---

## 6. Division of work

This is a solo final assignment (Le Dinh Minh Quan, 23127460). "Division of work" is therefore organized by workstream
to make sequencing, reuse, and the offline→Colab handoff explicit, not by person.

| Workstream | Scope | Reuse vs new | Milestones |
|------------|-------|--------------|------------|
| **Research & licensing** | Hub id verification, license flags, reuse map | reuse research process; new license audit for SLT | M0 |
| **Pose & data** | keypoint layout, lexicon, synthetic generator, SeedPoseEngine | **new** (pose layout + trajectory generator) | M1, M2 |
| **Segmentation & recognition** | motion segmenter, numpy centroid recognizer, transformer recognizer | **new** (segmenter + pose recognizer) | M3, M4, M6 |
| **Translation & eval** | gloss→text translator, BLEU/chrF/WER/ROUGE, baselines, robustness sweep | **reuse** P13/P14 MT metrics + seq2seq pattern + `t5-small`/`m2m100` | M5, M6, M8 |
| **Agent** | 5-decision FSM, confidence gating, verify gate, abstention, advisory LLM brain | **new** (sign-translation agent) over reused FSM scaffolding | M7 |
| **Infra & deploy** | config/logging/registry/autoreport/charts/monitoring/automation/grading/cli/api; FastAPI + Gradio + Docker + HF Space | **reuse** P15/P17/P19/P20 templates; new Docker deps (mediapipe/ffmpeg/libGL) | M8, M9 |
| **Docs & ethics** | 14 Section-I docs incl. the metric-reliability caveat, model card, ethics, privacy/robustness | reuse doc template; new SLT-specific content | M0, M8, M9 |

**Suggested timeboxing if the schedule compresses:** protect M2–M5 (the offline spine and its metrics) above all — a
fully-green CPU pipeline with honest synthetic numbers is a complete, defensible deliverable. M6 (neural Colab upgrade)
is the highest-value stretch; M9 deploy artifacts and the optional LLM brain are the first items to cut.

---

## 7. Reuse summary

- **From P13/P14 (MT):** BLEU/chrF/WER metric implementations, the seq2seq train/eval pattern, and the `t5-small` /
  `m2m100_418M` backbones — the gloss→text translator is a standard MT problem ("treat the recognized symbolic
  sequence as a source language", the `sign/sockeye-signwriting-to-text` precedent).
- **From P15/P17/P19/P20:** the embedded-gold synthetic-generator + Seed/Stub offline pattern, and the standard
  config / logging / registry / autoreport / charts / monitoring / automation / grading / cli / api templates.
- **New for P01:** the pose-keypoint layout + front-end (MediaPipe Holistic / SeedPoseEngine), the synthetic
  **pose-sequence** generator (motion trajectories, not images), the **motion-based segmenter**, the per-segment
  **gloss recognizer** (numpy nearest-centroid offline + transformer on Colab), and the **sign-translation agent**.

---

## 8. Acceptance checklist (maps to definition of done)

- [ ] Offline spine runs on CPU with no MediaPipe / torch / video / network (D1).
- [ ] Held-out synthetic split reports recognition (gloss-WER, position-aligned accuracy, seq-exact-match),
      translation (BLEU-1..4, chrF/chrF++, ROUGE-L, WER), segmentation (boundary-F1), and abstention rate (D2).
- [ ] All four baselines report; trained core beats identity/most-frequent/random and approaches the Seed oracle (D2).
- [ ] Verified offline targets reproduced: clean gloss accuracy 1.0, BLEU 99+, segmentation-F1 1.0 vs most-frequent
      floor ~0.02 and identity-translate BLEU ~84 (lexicon adds ~15 BLEU); pose-noise degrades recognition; pure
      noise abstains (D2/D3).
- [ ] Agent's 5 decisions each fire in a traced test; pure-noise input abstains (D3).
- [ ] Colab run reproduces headline numbers with MediaPipe + transformer recognizer + `t5-small`; real-data smoke
      test passes on the Icelandic slice (D4).
- [ ] FastAPI `POST /translate`, Gradio demo, Docker image, HF Space all ship; offline CPU path works (D5).
- [ ] All 14 Section-I docs complete; SLT-metric reliability caveat surfaced in `translation_evaluation`, autoreport,
      and model card; every restrictive dataset/model license flagged (D6).
