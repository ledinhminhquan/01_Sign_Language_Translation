# P01 — Sign Language Translation: DESIGN BRIEF (LOCKED)

**Author:** Le Dinh Minh Quan (23127460) · HF Pro `ledinhminhquan` · Colab Pro+ (H100, auto-adapt A100/L4/T4)
**Status:** LAST + HARDEST project of the NLP-in-Industry final assignment. First **video/pose → text sequence** task (prior 18 were text/OCR).
**This document is authoritative.** It LOCKS the stack and design. All HF ids below were verified to EXIST via the HF Hub tools by the four finders; licenses are flagged loudly. Where an id could not be confirmed, that is stated explicitly — **do NOT invent ids.**

---

## 0. The one-paragraph thesis

Sign Language Translation (SLT) maps a sign-language **video** (or its **pose/keypoint sequence**) → spoken-language **text**, optionally via an intermediate **gloss**. The defining reality of this project is **licensing**: *every* continuous Sign→Text corpus with glosses on the Hugging Face Hub is non-commercial, "other"/research-only, gated, or unspecified — and *no* production-grade, drop-in Sign-Video→Sentence or Gloss→Text seq2seq checkpoint exists on the Hub. Therefore the architecture is a **cascade** whose front-end is **pretrained/algorithmic and never trained** (MediaPipe Holistic pose layout, with ViTPose++ as the permissive HF body extractor for the optional real path), and whose **single trainable, measured component** is a small **pose/gloss-sequence → text seq2seq** (`t5-small` default). The **primary offline data is a deterministic SYNTHETIC pose-sequence generator** that embeds the gold gloss/text and is read by a Stub/Seed recognizer — so the entire pipeline (ingest → segment → recognize → translate → eval → agent → tests) runs with **no mediapipe / torch / video / network**, exactly mirroring the P15/P17/P19/P20 offline spines. This is a defensible, honest design forced by the data licensing, not a shortcut.

---

## 1. Problem statement + cascade architecture

### 1.1 Task
- **Input:** a sign-language clip represented as a **pose/keypoint sequence** `(T, K, C)` per the MediaPipe Holistic layout (front-end output), or — offline — the synthetic generator's deterministic trajectories.
- **Output:** spoken-language **text** (default English-like / German-like target), optionally via an intermediate **gloss** sequence.
- **Framing (LOCKED): Sign2Gloss2Text cascade** as the primary mode (mirrors the P13–P20 cascades, gives a clean offline spine and a gloss-WER recognition metric), with **direct Sign2Text** supported as a secondary mode via config. Gloss intermediate is the classic PHOENIX recipe (Camgöz et al.).

### 1.2 The cascade (3 stages)

```
            PRETRAINED / ALGORITHMIC                 TRAINABLE (the only measured unit)
            (never trained, no license risk)
  ┌───────────────────────────────────┐   ┌──────────────────────────────────────────────┐
  │ FRONT-END: pose/keypoint extractor │   │ CORE: pose/gloss-sequence → text seq2seq      │
  │  • MediaPipe Holistic (Apache-2.0, │   │  • encoder over pose tokens  → gloss (CSLR/CTC)│
  │    off-HF) = the SCHEMA target     │──▶│  • seq2seq (t5-small) gloss → text             │
  │  • ViTPose++ (Apache, HF) optional │   │  • OR direct pose-token → text (Sign2Text)     │
  │    real-path body extractor        │   │                                                │
  │  • OFFLINE: synthetic generator    │   └──────────────────────────────────────────────┘
  │    emits the SAME (T,K,C) tensors  │                         │
  └───────────────────────────────────┘                         ▼
                                                    BLEU-1..4 / chrF / ROUGE-L (text)
                                                    WER (gloss recognition)
```

- **Stage A — Front-end (pretrained/algorithmic, NOT trained):** produces a per-frame keypoint tensor. Online/real path = MediaPipe Holistic (hands+body+face) and/or ViTPose++ for body. Offline path = synthetic generator emitting **byte-compatible** tensors so downstream code never branches on data source.
- **Stage B — Recognizer (Sign2Gloss / CSLR):** maps the pose sequence → an ordered **gloss** token stream. In production this is a small pose+CTC model (architecture reference: `manohonsy/how2sign-pose-cslr`, MIT, 4.8M params). Offline this is the **Stub/Seed recognizer** that reads the embedded gold gloss.
- **Stage C — Translator (Gloss2Text, the trainable core):** a plain seq2seq MT model (`t5-small`) mapping gloss/pose-token sequence → spoken text. **This is the single trainable, measured component**, exactly matching the P13/P14 BLEU/chrF/WER tooling.

**Why pose, not video features, as the schematic spine:** keypoints are low-dimensional, interpretable, CPU-runnable, license-clean (MediaPipe is Apache-2.0), and **trivially synthesizable** as smooth trajectories. A frozen video encoder emits an opaque 768-d clip embedding that the synthetic generator cannot cheaply fabricate — and every canonical video encoder is **CC-BY-NC-4.0 (non-commercial)**. The video-encoder route is therefore **rejected** for the front-end and kept only as a documented, flagged alternative.

---

## 2. VERIFIED model stack (exact HF ids + licenses)

### 2.1 Trainable core (LOCKED) — all PERMISSIVE, all confirmed

| Role | HF id | License | Params | Why |
|---|---|---|---|---|
| **PRIMARY core** | `google-t5/t5-small` | **Apache-2.0** ✅ | 60.5M | Default. Tiny, fast on T4/L4, native `translation` task, trivial gloss→text. YouTube-SL-25 SOTA uses T5. |
| Byte-level variant | `google/byt5-small` | **Apache-2.0** ✅ | ~300M | Tokenizer-free → robust to OOV glosses, fingerspelling, FSW/pose-token vocabularies. Use when source tokens are non-word symbols. |
| Heavier / multilingual (P13/P14 REUSE) | `facebook/m2m100_418M` | **MIT** ✅ | 418M | Drop-in reuse of the existing P13/P14 MT harness; multilingual targets. |
| Optional MT comparators | `facebook/mbart-large-50` (MIT), `Helsinki-NLP/opus-mt-de-en` (Apache-2.0) | permissive ✅ | — | mBART for SOTA text2gloss parity; opus-mt-de-en if mimicking German PHOENIX targets. |

**Decision:** train `google-t5/t5-small` as the default measured core; expose `byt5-small` and `m2m100_418M` via config. Frame the task as **Gloss→Text** so the core is plain seq2seq MT.

### 2.2 Front-end models (pretrained, NOT trained)

| Role | HF id | License | Notes |
|---|---|---|---|
| **Body extractor (real path, RECOMMENDED)** | `usyd-community/vitpose-plus-base` | **Apache-2.0** ✅ | 125.4M, `VitPoseForPoseEstimation` in `transformers`. COCO-17 / up to 133 whole-body. Needs an upstream person detector (RT-DETR / `AutoModelForObjectDetection`). |
| Lighter ViTPose | `usyd-community/vitpose-plus-small`, `usyd-community/vitpose-base-simple` | **Apache-2.0** ✅ | Lightest trainable-free options. |
| **Hands (algorithmic, off-HF)** | `mediapipe` (PyPI) — Holistic / Tasks `HandLandmarker`,`PoseLandmarker`,`FaceLandmarker` | **Apache-2.0** ✅ | **NOT on HF.** The canonical 21×2 finger keypoints; pairs with ViTPose for fingers. The **schema target** the synthetic generator mimics. |
| Frozen video fallback (permissive) | `microsoft/xclip-base-patch32` | **MIT** ✅ | 196.6M. Only if going video→features instead of pose. |

### 2.3 Real-SLT checkpoints — references ONLY (none drop-in)

| HF id | License | Use |
|---|---|---|
| `manohonsy/how2sign-pose-cslr` | **MIT** ✅ | **Architecture reference** for the Stage-B pose+CTC recognizer (4.8M params — proves student-scale CSLR is sane). Not a finished translator. |
| `sign/sockeye-signwriting-to-text` | **MIT** ✅ | SignWriting→text precedent (Sockeye format, NOT `transformers` — cannot `from_pretrained`). Reuse the **idea** (symbolic seq as source language → standard MT) + BLEU/chrF recipe, not the weights. |
| `sign/signwriting-clip` | **UNSPECIFIED** 🚩 | Frozen embedding reference only; treat as all-rights-reserved. |

### 2.4 🚩 MODEL LICENSE FLAGS (loud)

- **NON-COMMERCIAL (CC-BY-NC / -SA):** `MCG-NJU/videomae-base`(+`-finetuned-kinetics`), `facebook/timesformer-base-finetuned-k400`, all `OpenGVLab/VideoMAEv2-*` (also need `trust_remote_code`), `sign/mediapipe-vq` (NC-SA, the "pose→discrete tokens" reference — **avoid**), `sign/sockeye-text-to-factored-signwriting`, `ihsanahakiim/videomae-base-finetuned-signlanguage*`. **→ Do NOT use any of these in the clean pipeline.**
- **UNSPECIFIED / all-rights-reserved (do NOT assume reusable):** `sign/signwriting-clip`, `sign/signwriting-transcription`, all `PhoenixHu/grpo_internvl2_5_how2sign_1b_*` fine-tunes, `yunghee/ViTPoseCompression`, most 0-download personal repos.
- **TOO HEAVY / license-murky (context only):** `OpenGVLab/InternVL2_5-1B` (base MIT, 938M) and its How2Sign GRPO fine-tunes — 1B vision-LLM, end-to-end, unlicensed fine-tunes. SOTA-direction context only.
- **PERMISSIVE / SAFE (use these):** `google-t5/t5-small`, `google/byt5-small`, `facebook/m2m100_418M`, `facebook/mbart-large-50`, `Helsinki-NLP/opus-mt-de-en`, `microsoft/xclip-base-patch32`, all `usyd-community/vitpose-*`, `manohonsy/how2sign-pose-cslr`, `sign/sockeye-signwriting-to-text`, `mediapipe` (PyPI).

---

## 3. VERIFIED dataset stack (exact HF ids + licenses + schemas)

**Headline:** No permissively-licensed, ready-to-train, pose+text **continuous** SLT corpus exists on HF. Every real corpus is NC, "other", gated, or unspecified. Real datasets are wired in as **optional secondary loaders behind license gates**; the synthetic generator is the **primary** data path.

### 3.1 Continuous SLT (Sign→Text capable) — all license-encumbered

| HF id | Content / schema | License | FLAG |
|---|---|---|---|
| `aipieces/RWTH-PHOENIX-Weather-2014T` ⭐ | German SL. Video (`large_binary`) + **`orth`=GLOSS** + **`translation`=German text**. Splits train/dev/test (RGB ~632/43/50 MB). The ONLY confirmed HF set with gloss AND text aligned. | **UNSPECIFIED** (upstream PHOENIX-2014T = CC-BY-NC-SA-4.0) | 🚩 **NC / research-only.** Video, not pre-extracted pose (run MediaPipe yourself). |
| `lukasbraach/rwth_phoenix_weather_2014` | PHOENIX-2014 **recognition** (gloss only, no text). Schema `id`,`transcription`(gloss),`frames`. Configs multisigner/pre-training/signerindependent. | **CC-BY-NC-4.0** | 🚩 **NON-COMMERCIAL.** Gloss-only; viewer disabled (loading script). |
| `aipieces/How2Sign` | ASL→English. Video (`large_binary`) + `SENTENCE` (English). **No gloss.** ~30.7/1.65/2.24 GB. | **UNSPECIFIED** (upstream CC-BY-NC-4.0) | 🚩 **unspecified / NC.** Heavy video. |
| `PSewmuthu/How2Sign_Holistic` ⭐ | **MediaPipe Holistic** landmarks (.npy: pose+face+hands) per sentence clip + English text. **Best pre-extracted pose+text.** Load via `huggingface_hub` file download (viewer can't parse .npy). | **MIT** (on files) | ✅ tag / 🚩 **underlying How2Sign is CC-BY-NC-4.0** — MIT likely covers only extraction format. Treat signs as NC. |
| `Exploration-Lab/iSign` | Indian SL; defines **SignPose2Text** (your exact task). 118K+ pairs. | **CC-BY-NC-SA-4.0** + 🔒 GATED | 🚩 **NC + ShareAlike + gated.** |

### 3.2 Pre-extracted pose / features (secondary, mostly word-level)

| HF id | Content / schema | License | FLAG |
|---|---|---|---|
| `Kibalama/poseformer-sign-language` | WLASL→MediaPipe `landmarks` (3D nested float = pose+face+hands), `label`. 12K rows. Largest Kibalama set. | UNSPECIFIED (WLASL "other") | 🚩 unspecified. **Word-level**, not continuous. |
| `Kibalama/wlasl-processed-motion-tokens` | `poses` (3D [frames][joints][xyz]), `label`, velocity, acceleration. 1K rows. Cleanest pose tensors. | UNSPECIFIED | 🚩 unspecified. Word-level. |
| `Kibalama/wlasl-upper-arm-pose-landmarks` | `video_id`,`poses`(T×K×3),`label`. 1K rows. | UNSPECIFIED | 🚩 unspecified. Word-level. |
| `FangSen9000/How2Sign-dwpose` | DWPose keypoints (pose tensors in repo files; viewer shows only `text`). | **MIT** (derived) | ✅ tag / 🚩 NC source. |
| `sodonne6/how2sign-resnet50-mediapipe-30-pose` | Pre-extracted ResNet50 + MediaPipe 30-pose features (.tar.zst). Closest to "frozen-encoder features → seq2seq". | **CC-BY-NC-4.0** | 🚩 **NON-COMMERCIAL.** |
| `Shakibyzn/phoenix14t-features` | Pre-extracted I3D/S3D/CLIP features for PHOENIX14T. | MIT (files) / DGS source NC | 🚩 source NC. |
| `Voxel51/WLASL` | Canonical WLASL, 11,980 isolated-sign videos (recognition). Needs FiftyOne; viewer broken. | **other** | 🚩 restrictive WLASL terms. |

### 3.3 Permissive smoke-test data only (tiny / isolated — NEVER the main benchmark)

| HF id | Content | License |
|---|---|---|
| `Sigurdur/icelandic-sign-language` | YouTube-SL-25 slice. **video_id + timed transcript only (no media/pose)** — you'd scrape YouTube. 214 rows. | **Apache-2.0** ✅ |
| `om192006/sign_language_keypoints` | Pre-extracted MediaPipe keypoints, 29 isolated gestures, LSTM-ready. Good pose-schema template. | **MIT** ✅ |
| `merterm/intensified-phoenix-14-t` | Text-side PHOENIX-14T augmentation (gloss/text only, no pose/video). | **MIT** ✅ |

### 3.4 NOT FOUND on HF (do NOT invent ids)
MS-ASL, BOBSL/BSL, WMT-SLT, official **PHOENIX-2014T clean/permissive mirror**, CSL-Daily, first-party YouTube-ASL / YouTube-SL-25, official standalone **I3D / S3D** weight repos, `Exploration-Lab/ISLTranslate`. AUTSL exists only as `aipieces/AUTSL` (video, unspecified, recognition not translation). `pedroodb/glosl-*` (GloSL unified-pose effort) is a **structural reference** but PHOENIX14T copy is NC-SA + **private** and LSA-T data is "unlicensed."

### 3.5 Dataset decision (LOCKED)
- **Primary data = synthetic generator (§4).** Non-negotiable.
- **Optional real loaders behind a `license_ack` gate, default OFF:** `aipieces/RWTH-PHOENIX-Weather-2014T` (gloss+text target) and `PSewmuthu/How2Sign_Holistic` (pre-extracted pose+text). Both flagged effectively NC.
- **Smoke-test loader validation:** `om192006/sign_language_keypoints` (MIT) and `merterm/intensified-phoenix-14-t` (MIT) only — never as the measured benchmark.

---

## 4. OFFLINE SYNTHETIC pose-sequence generator (the spine)

The spine that lets segment → recognize → translate → eval → agent → tests run with **no mediapipe / torch / video / network**. Mirrors P15/P17/P19/P20. Pure-Python + numpy (numpy optional; can fall back to lists).

### 4.1 Keypoint tensor (MediaPipe-compatible, so downstream never branches)
- **Shape `(T, K, C)`**, coordinates normalized to `[0,1]` image space (MediaPipe convention).
- **Minimal SLT layout (default): `K = 75`** = 25-pt upper-body pose subset + 21 left hand + 21 right hand. `C = 3` (x,y,z) or `4` (+visibility).
- **Optional holistic-full layout: `K = 543`** = 33 pose + 21 + 21 + 468 face, to byte-match `PSewmuthu/How2Sign_Holistic` exactly.
- Layout constants and index ranges fixed in `data/pose_layout.py` so synthetic and real tensors are interchangeable.

### 4.2 Deterministic trajectory synthesis (per sign in a small vocab)
- A small fixed **vocabulary** (e.g. 20–60 glosses) each mapped to:
  - a **canonical hand path** = a parametric motion primitive (line / arc / circle / oscillation) over a fixed frame count,
  - a fixed **start/end handshape** (a canonical 21-pt finger configuration), and
  - a **wrist trajectory**.
- **Sentence composition:** concatenate per-gloss segments + short **movement-epenthesis (transition) frames** between signs.
- **Determinism:** seed the RNG from the **gloss id** → fully reproducible. Light augmentation = small Gaussian jitter + global affine (scale/translate), all deterministic under a fixed master seed.
- **Embedded gold:** each sample carries `(pose_tensor, gold_gloss_sequence, gold_text)`. A deterministic **gloss↔text reordering rule** (sign order ≠ spoken order) makes the identity baseline genuinely weak — proving the trainable core must learn reordering.

### 4.3 Stub/Seed recognizer (Stage-B offline)
- Reads the **embedded gold gloss** directly (or trivially decodes the deterministic handshape codes back to gloss ids). No model, no torch.
- Gives an **oracle = perfect recognition**, isolating the trainable **translate** stage as the single measured thing — exactly the P15/P17/P19/P20 pattern.
- A configurable **noise knob** (drop/insert/substitute gloss tokens at a fixed rate) lets you simulate an imperfect recognizer to test the agent's abstain logic and the gloss-WER metric end-to-end.

### 4.4 Why this is mandatory (not a shortcut)
Justified by §3: no permissive continuous pose+text corpus exists; the trainable core must be measured on data we fully control and can run anywhere with zero heavy deps. The synthetic spine makes the seq2seq core the **only** trainable, measured unit.

---

## 5. Metric set + baselines (reuse P13/P14)

### 5.1 Metrics — single `metrics/translation.py`, `(hyps, refs)` lists in, no new infra

| Metric | Stage | Tool (P13/P14 reuse) | Dir |
|---|---|---|---|
| **BLEU-1, -2, -3, -4** (BLEU-4 = headline) | Translation (text) | `sacrebleu` (`effective_order=True` for short seqs) | ↑ |
| **chrF / chrF++** | Translation (text) | `sacrebleu` native | ↑ |
| **ROUGE-L** | Translation (text) | `rouge_score` / `evaluate` | ↑ |
| **WER over glosses** | Recognition (CSLR) | `jiwer` (reuse existing WER; gloss tokens = "words") | ↓ |
| **BLEURT / COMET** (optional) | Translation (semantic) | `evaluate` | ↑ |
| Top-1 / Top-5 accuracy (if isolated-sign mode) | Recognition | sklearn | ↑ |

**Convention (replicate PHOENIX tabulation):** report BLEU-1..4 + chrF + ROUGE-L (translation) and gloss-WER (recognition), on **dev and test separately**.

**🚩 Metric-honesty caveat (cite in docs/autoreport):** Yazdani et al. (Oct 2025, *A Critical Study of Automatic Evaluation in SLT*, hf.co/papers/2510.25434) show BLEU/chrF/ROUGE/BLEURT are unreliable for SLT — length-sensitive, blind to hallucinations and semantic equivalence. Report the standard set **and** flag these limitations (matches the honesty stance of prior projects).

### 5.2 Baselines (isolate the trainable core, à la P15/P19/P20)
1. **Identity / source-passthrough** — emit input gloss as "translation." Lower bound; exposes gloss↔text reordering the model must learn (deliberately weak via §4.2).
2. **Most-frequent-sentence** — always emit the most frequent training target. Degenerate BLEU lower bound.
3. **Gloss → most-frequent-word dictionary** — per-gloss → most-frequent aligned spoken word (unigram baseline).
4. **Stub/Seed oracle (perfect recognition)** — upper bound for the translate stage in isolation.
5. **Trainable core** (`t5-small`) — must beat 1–3, lower-bounded by oracle (4).

---

## 6. The 5-decision AGENT design

A deterministic decision agent over the cascade. Each decision **gates on an explicit intermediate signal**; low-confidence/OOV triggers **abstain**.

| # | Decision | Gating signal (intermediate) | Action / abstain rule |
|---|---|---|---|
| **D1** | **Ingest & validate** | pose-tensor shape `(T,K,C)` matches `pose_layout`; `T ≥ min_frames`; not all-zero/NaN | If malformed/empty → **abstain** ("uninterpretable input"). |
| **D2** | **Detect / segment signs** | per-frame motion energy / epenthesis gaps → candidate sign boundaries; segment count | If no stable segments (motionless/over-jittered) → **abstain** ("no signs detected"). |
| **D3** | **Recognize gloss (Sign2Gloss)** | per-gloss recognizer confidence; **OOV check** vs known vocab | If max conf `< τ_gloss` **or** OOV rate `> ρ` → **abstain on that segment** / flag low-confidence gloss. |
| **D4** | **Translate text (Gloss2Text)** | seq2seq decoder score / avg token logprob; length-ratio sanity vs source | If decode score `< τ_text` or degenerate length ratio → **abstain** ("low-confidence translation") and emit gloss-only output. |
| **D5** | **Abstain / emit** | aggregate confidence across D2–D4 | Emit text **iff** all gates pass; else return best partial (gloss sequence) + explicit abstain reason + confidence breakdown. |

- **Offline-driven:** in the synthetic spine the Stub recognizer's noise knob (§4.3) lets every gate (especially D3 OOV and D4 low-confidence) be exercised deterministically in tests.
- **Honesty-first:** abstain reasons and per-decision confidences are surfaced in the autoreport, consistent with prior projects.

---

## 7. The 10 Section-I documentation topics

1. **Problem & framing** — SLT defined; Sign2Gloss2Text vs Sign2Text; why pose-input (SignPose2Text) is the trainable core; relation to the P13–P20 cascade lineage.
2. **Cascade architecture** — front-end (pretrained/algorithmic) → recognizer → trainable seq2seq translator; data-flow `(T,K,C)` → gloss → text.
3. **Pose front-end** — MediaPipe Holistic layout (33+21+21+468 = 543; SLT-reduced 75); ViTPose++ permissive HF body extractor; why finger keypoints are linguistically load-bearing; rejection of NC video encoders.
4. **Synthetic offline spine** — deterministic trajectory generator, embedded gold, Stub/Seed recognizer; why it is primary data (license scarcity); P15/P17/P19/P20 mirroring.
5. **Datasets & the licensing reality** — every continuous-SLT corpus is NC/other/gated/unspecified; the verified id table + flags; optional gated real loaders.
6. **Model stack** — verified permissive cores (t5-small/byt5-small/m2m100); reference-only SLT checkpoints; full license-flag table.
7. **Training the seq2seq core** — Gloss→Text as plain MT; config-driven backbone swap; Colab tier auto-adaptation (H100/A100/L4/T4); reuse of P13/P14 training harness.
8. **Metrics & evaluation** — BLEU-1..4 / chrF / ROUGE-L (text) + gloss-WER (recognition); dev/test convention; **metric-unreliability caveat (Yazdani 2025)**.
9. **Baselines & the agent** — 4 baselines + oracle; the 5-decision agent and its abstain gates; how the synthetic noise knob exercises them.
10. **Limitations, ethics & licensing honesty** — synthetic ≠ real signing; signer-independence, gloss scarcity, continuous-vs-isolated gaps; NC-data constraints; metric blindness to hallucination; responsible-use statement.

---

## 8. REUSE map (from earlier projects) + what is NEW for P01

### 8.1 REUSE (lift with minimal change)
- **P13/P14 MT metrics** → `metrics/translation.py`: `sacrebleu` (BLEU-1..4, chrF/chrF++), `rouge_score` (ROUGE-L), `jiwer` (WER) — gloss tokens treated as words. **No new metric infra.**
- **P13/P14 seq2seq training/eval harness** → train the `t5-small` / `m2m100_418M` core; `m2m100` slots directly into the existing harness.
- **P15/P17/P19/P20 offline spine pattern** → synthetic generator + embedded gold + Stub/Seed recognizer; trivial-baseline-isolates-core methodology; deterministic, dep-free tests.
- **Standard templates** (all projects) → config / logging / registry / autoreport / monitoring / automation / grading / CLI / API — reused verbatim, with the SLT metric/agent plugged in.

### 8.2 NEW for P01 (first time in the assignment)
- **Pose/keypoint modality** — `(T,K,C)` tensors, `pose_layout.py`, MediaPipe Holistic schema; first non-text input.
- **Synthetic POSE-SEQUENCE generator** — deterministic hand/wrist trajectories + handshapes + epenthesis (vs prior text/OCR synthetic spines).
- **Two-stage Sign2Gloss2Text cascade with a CSLR recognizer stage** and a **gloss-WER** recognition metric alongside translation metrics.
- **Pretrained pose front-end** (MediaPipe / ViTPose++) as a non-trained component, plus an explicit **rejection of NC video encoders** on weight/opacity/license grounds.
- **License-gated real-data loaders** — because the entire corpus space is NC/other/gated/unspecified, a hard `license_ack` gate (default OFF) is a new, project-specific safeguard.

---

## 9. LOCKED decisions (quick reference)

- **Front-end:** MediaPipe Holistic layout (Apache-2.0, off-HF) as schema; `usyd-community/vitpose-plus-base` (Apache-2.0) for the optional real body path. **Reject** all CC-BY-NC video encoders and `sign/mediapipe-vq` (NC).
- **Trainable core:** `google-t5/t5-small` (Apache-2.0) default; `google/byt5-small` (Apache-2.0) + `facebook/m2m100_418M` (MIT) via config.
- **Recognizer reference:** `manohonsy/how2sign-pose-cslr` (MIT) — architecture only.
- **Primary data:** SYNTHETIC deterministic pose generator + Stub/Seed oracle. Real loaders (`aipieces/RWTH-PHOENIX-Weather-2014T`, `PSewmuthu/How2Sign_Holistic`) are **optional, gated, flagged effectively NC**.
- **Metrics:** BLEU-1..4 + chrF + ROUGE-L (text), gloss-WER (recognition), dev/test separate, with the Yazdani-2025 reliability caveat.
- **Mode:** Sign2Gloss2Text primary; Sign2Text secondary.
- **Confirmed-id quick list:** `google-t5/t5-small`, `google/byt5-small`, `facebook/m2m100_418M`, `facebook/mbart-large-50`, `Helsinki-NLP/opus-mt-de-en`, `usyd-community/vitpose-plus-base`, `microsoft/xclip-base-patch32`, `manohonsy/how2sign-pose-cslr`, `sign/sockeye-signwriting-to-text`, `aipieces/RWTH-PHOENIX-Weather-2014T`, `PSewmuthu/How2Sign_Holistic`, `om192006/sign_language_keypoints`, `merterm/intensified-phoenix-14-t`. **Do NOT invent ids beyond these.**
