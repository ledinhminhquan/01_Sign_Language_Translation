# P01 — Sign Language Translation · Model Selection

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment (the last & hardest project).
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`.

This document justifies every model in the P01 stack. The defining finding from the design research is structural and
dictates the whole selection: **there is no permissively-licensed, directly-loadable Sign→Text or Gloss→Text seq2seq
checkpoint on the Hugging Face Hub, and no permissive continuous-SLT corpus either.** That is not a gap to be patched —
it *is* the field. So instead of "download a Sign→Text model and fine-tune it" (the pattern that worked for P02–P20),
P01 selects a **frozen, algorithmic pose front-end** plus a **small seq2seq core we train ourselves** from a permissive
backbone, driven offline by a synthetic pose-sequence generator. Each choice below is made against three hard filters:

1. **License must be commercially clean** (Apache-2.0 / MIT). Every non-commercial (CC-BY-NC*), gated, or
   unspecified/all-rights-reserved asset is explicitly *flagged and avoided* as a load-bearing dependency.
2. **Student-scale** — must train or run on a single Colab T4 (16 GB) and run offline on CPU with no network.
3. **Honest measurability** — the trained core must be a single, clearly-measured unit, not a black box hiding behind a
   pretrained checkpoint we did not actually train.

---

## 0. The cascade and where each model sits

```
VIDEO ──(MediaPipe Holistic, FROZEN / algorithmic — NOT trained)──► pose-keypoint sequence
        ──► motion-based SEGMENT into sign units
        ──► RECOGNIZE gloss per segment        ◄── TRAINABLE core #1 (pose → gloss)
        ──► TRANSLATE gloss → spoken text       ◄── TRAINABLE core #2 (gloss → text seq2seq)
        ──► assemble sentence
```

There are exactly **two model families to select**, plus the front-end:

| Stage | Component | Trained? | Selected model |
|---|---|---|---|
| Front-end | video → pose keypoints | **No** (pretrained / algorithmic) | **MediaPipe Holistic** (Colab) / **SeedPoseEngine** (offline) |
| Core #1 | pose segment → gloss | **Yes** | compact **transformer encoder** (Colab) / **numpy nearest-centroid** (offline) |
| Core #2 | gloss → spoken text | **Yes** | **`google-t5/t5-small`** (default) |

The segmenter (D2) is a deterministic motion/velocity algorithm, not a learned model, so it is covered in
`architecture.md` / `agent_architecture.md`, not here.

---

## 1. Front-end: the frozen pose extractor

The front-end turns raw RGB video into a per-frame **pose-keypoint vector**. It is deliberately **not trained** — it is a
pretrained, algorithmic feature extractor, frozen exactly like the way prior projects used a pretrained tokenizer/OCR
front-end and only trained the downstream head.

### 1.1 Selected: MediaPipe Holistic (Google, Apache-2.0)

**Decision: MediaPipe Holistic is the production front-end on Colab; the offline `SeedPoseEngine` is its stand-in.**

| Property | Value |
|---|---|
| Provider | Google |
| License | **Apache-2.0** (clean for commercial use) |
| What it emits | hand + body (+ face) landmarks per frame, per-frame |
| Trained in P01? | **No** — frozen, algorithmic |
| Output layout (`pose/layout.py`) | **2×21 hand + 25 body landmarks × 3 coords** per frame |

Justification:

- **License is clean.** Apache-2.0 imposes no non-commercial restriction, no ShareAlike, no gating. This is the single
  most important property: it lets the deployed system run a pose extractor without inheriting an NC license from the
  front-end (contrast §1.3).
- **It is the de-facto standard layout for SLT pose corpora.** The Hub datasets that store sign-language pose data —
  `PSewmuthu/How2Sign_Holistic` (MediaPipe Holistic landmark `.npy`) and `om192006/sign_language_keypoints`
  (pre-extracted MediaPipe keypoints) — are MediaPipe Holistic outputs. Selecting the same front-end means our keypoint
  layout in `pose/layout.py` is *schema-compatible* with real data: the synthetic generator emits the same
  `2×21 + 25` landmark shape, so the trained recognizer transfers from synthetic to real pose tensors without a
  reshaping layer.
- **It is privacy-appropriate.** Pose extraction is the natural place to discard raw video and keep only abstract
  landmarks. Running MediaPipe on-device/edge and retaining only keypoints is the architecture the ethics/privacy
  posture requires (sign-language video is biometric + identifying — see `privacy_robustness.md`). A front-end that
  *needs* the raw frames sent to a server would undermine that posture.
- **It is free and offline-capable.** No API key, no per-call cost, runs on CPU. That keeps the Docker image
  (mediapipe + ffmpeg + libGL) self-contained and the demo runnable without network.

**Offline counterpart — `SeedPoseEngine`.** Offline (no mediapipe, no video, no GPU) the front-end role is played by the
`SeedPoseEngine`, which reads the **gold pose/gloss/text spec embedded on the synthetic sequence** by `data/synth_pose.py`
and returns the keypoint sequence in the identical `2×21 + 25` layout. This is the same Seed/Stub pattern used in
P15/P17/P19/P20: a deterministic offline engine that lets the entire pipeline (segment → recognize → translate → eval →
agent → tests) run with zero heavy dependencies, then swaps cleanly for MediaPipe on Colab.

### 1.2 Permissive video-encoder alternative: `microsoft/xclip-base-patch32` (MIT)

If a future variant chooses to go **video → features directly** (skipping explicit pose landmarks — a "Sign2Text" mode
rather than the pose cascade), the selected permissive video encoder is **`microsoft/xclip-base-patch32`**.

- **License: MIT** — clean, commercial-safe.
- It is a video CLIP encoder: produces a pooled clip embedding usable as a frozen feature for the downstream recognizer.
- It is listed as the *alternative* and not the default precisely because it loses the pose cascade's advantages:
  per-frame landmarks give us a motion signal we can segment on (D2) and an interpretable per-sign confidence; a single
  pooled video embedding does not. Pose keypoints are also far cheaper to store, transmit, and anonymize than video
  features. So x-clip is the documented escape hatch, not the primary path.

### 1.3 Rejected front-ends (license-disqualified)

| id | License | Why rejected |
|---|---|---|
| `MCG-NJU/videomae-base` | **CC-BY-NC** | **Non-commercial.** A strong masked-video encoder, but the NC clause would poison the whole pipeline's license. **AVOID.** |
| `sign/mediapipe-vq` | **CC-BY-NC-SA** | **Non-commercial + ShareAlike.** A sign-specific pose quantizer, but NC-SA is doubly disqualifying for a deployable system. **AVOID.** |

The lesson generalizes: **the best-known video backbones in this space are non-commercial.** Choosing the algorithmic,
Apache-2.0 MediaPipe front-end is what keeps P01 commercially clean end-to-end. We never let a CC-BY-NC* asset become a
load-bearing dependency.

---

## 2. Trainable core #1 — the pose-segment → gloss recognizer

This is the **trainable heart of the system** and the component the recognition metrics actually measure. It takes one
segmented sign (a short pose sub-sequence) and predicts a gloss with a confidence.

### 2.1 On Colab: a compact transformer encoder over pose frames

- A small **transformer encoder** over the per-frame keypoint vectors of a segment, with a gloss classification / CTC
  head. This is our own model — there is no checkpoint to download (see §4).
- **Scale reference — `manohonsy/how2sign-pose-cslr` (MIT, 4.8M params).** This Hub model is a pose + CTC continuous
  sign-language-recognition (CSLR) model trained on How2Sign, and it is the architectural proof point for the whole
  recognizer: it demonstrates that a **~5M-parameter** pose model is genuinely **student-scale** and trainable on a
  single T4. We treat it as an **architecture reference** (MIT-licensed, so we may read and mirror its design) — not as a
  checkpoint we load, because its vocabulary, gloss set, and training corpus are How2Sign-specific (NC upstream) and do
  not match our 40-gloss synthetic lexicon.
- Why ~5M and not larger: the recognizer's job is bounded — classify a single segment's motion trajectory into one of a
  small gloss vocabulary. A few transformer layers over a few-hundred-dim pose vector is ample, fits T4 comfortably, and
  keeps training reproducible within the assignment's compute budget.

### 2.2 Offline: a pure-numpy nearest-centroid classifier (no torch)

Offline the recognizer is a **pure-numpy nearest-centroid classifier** — and crucially **it genuinely classifies**: it
computes each segment's **mean-pose displacement** vector and assigns the gloss whose learned centroid is closest. This is
not a lookup of the embedded gold; it is a real (if simple) classifier over the pose geometry. The synthetic generator is
designed so this is *learnable*: each gloss has a fixed, deterministic motion direction in keypoint space, so the
per-sign mean displacement is a separable feature. The result is that the offline path:

- has **zero heavy dependencies** (no torch, no mediapipe, no GPU), runs on CPU in milliseconds;
- produces a real **confidence** (distance margin to the nearest centroid) that feeds the agent's D3 low-confidence gate
  and D5 abstention logic;
- **degrades honestly under noise** — the verified offline behavior is clean-input gloss accuracy = 1.0, dropping under a
  pose-noise robustness sweep, and **abstaining on pure-noise input** rather than emitting a confident wrong gloss.

This mirrors the offline/online split used across P15/P17/P19/P20: a faithful, dependency-free engine offline that the
heavyweight model replaces on Colab, both reading the same data layout.

---

## 3. Trainable core #2 — the gloss → text translator

Once glosses are recognized, the system **translates the gloss sequence into fluent spoken-language text**. This is a
standard text-to-text seq2seq problem, and it is the second trained unit. The translation is non-trivial because the
lexicon deliberately makes spoken text differ from gloss tokens (`THANK-YOU`→"thank you", `ME`→"i"), so the model must
learn lexical mapping and reordering — not copy.

### 3.1 Selected default: `google-t5/t5-small` (Apache-2.0)

| Property | Value |
|---|---|
| id | **`google-t5/t5-small`** |
| License | **Apache-2.0** (clean) |
| Params | **60.5M** |
| Fits | single Colab **T4 (16 GB)** with room to spare |
| Role | **PRIMARY** gloss → spoken-text seq2seq |

Justification:

- **License clean (Apache-2.0)**, tiny (60.5M → trains fast on T4), and a proven text-to-text workhorse. It is the
  smallest mainstream seq2seq that still produces fluent English, which matches the small-vocabulary, short-sentence
  nature of the gloss→text task.
- **Reuses the P13/P14 seq2seq train/eval harness** directly — same `transformers` Seq2SeqTrainer pattern, same
  BLEU/chrF/WER metric code. No new training infrastructure.
- The task framing is endorsed by precedent: **`sign/sockeye-signwriting-to-text` (MIT)** establishes the pattern of
  "treat the recognized symbolic sign sequence as a *source language* and run standard machine translation." We adopt
  exactly that framing — our recognized gloss sequence is the source language, t5-small is the MT model — but on the
  `transformers` stack rather than Sockeye, so it slots into the existing harness. (We cite Sockeye-SignWriting as
  precedent, not as a checkpoint we load.)

### 3.2 Documented alternatives

| id | License | When to prefer it |
|---|---|---|
| `facebook/m2m100_418M` | **MIT** | **Reuse from P13/P14.** Multilingual gloss→text if the target is a non-English spoken language; the MT harness already loads it. Larger (418M) — fine on T4 but slower. |
| `google/byt5-small` | **Apache-2.0** | **Byte-level**, so it is robust to **out-of-vocabulary glosses / symbolic pose tokens** with no subword-vocabulary mismatch — useful when gloss tokens contain unusual characters or when the source is treated as raw symbols. |

All three are commercially clean (Apache-2.0 / MIT). t5-small is the default because it is the smallest and reuses the
most existing code; m2m100 is the multilingual upgrade; byt5 is the symbolic-robustness upgrade.

---

## 4. Why there is no pretrained SLT checkpoint to use

The most important negative result in this selection — and the reason the trained core is *our own* small seq2seq rather
than a fine-tune of an existing Sign→Text model:

- **No permissive, directly-loadable Sign→Text / Gloss→Text seq2seq checkpoint exists on the Hub.** None load cleanly
  into `transformers` as a drop-in Sign→Text translator. The closest assets are either a different framework
  (`sign/sockeye-signwriting-to-text` is Sockeye, MIT — precedent, not a `transformers` checkpoint) or a recognizer, not
  a translator (`manohonsy/how2sign-pose-cslr` is a pose→gloss CSLR model, an architecture reference).
- **The translation-direction sign assets that do exist are non-commercial or all-rights-reserved:**

  | id | License | Status |
  |---|---|---|
  | `sign/sockeye-text-to-factored-signwriting` | **CC-BY-NC** | **AVOID** — non-commercial, and wrong direction (text→sign). |
  | `sign/signwriting-clip` | **unspecified** | **AVOID** — unspecified = all-rights-reserved by default. |
  | `PhoenixHu/grpo_internvl2_5_how2sign_*` | **unspecified** | **AVOID** — all-rights-reserved fine-tunes; not redistributable. |

- **The continuous-SLT corpora that would let you fine-tune such a model are themselves restrictive** (see
  `data_card.md`): `Exploration-Lab/iSign` (CC-BY-NC-SA + **gated** — it even defines *SignPose2Text*, our exact task),
  `aipieces/How2Sign` + `PSewmuthu/How2Sign_Holistic` (How2Sign is CC-BY-NC upstream), `Voxel51/WLASL` (license: other),
  `Kibalama/poseformer-sign-language` (WLASL-derived). The benchmark corpora (RWTH-PHOENIX-2014T, CSL-Daily, YouTube-ASL)
  are **not** redistributable Hub repos at all (academic licenses on university servers) — we do **not** invent ids for
  them.

**Consequence (and why this is the right design, not a compromise):** because no Sign→Text checkpoint and no permissive
continuous corpus is usable as a load-bearing dependency, the only honest, reproducible, commercially-clean path is to
**train a small seq2seq core ourselves** (numpy centroid + t5-small) on a **synthetic pose-sequence generator** that
embeds its own gold gloss/text. This makes the trained component genuinely *ours and genuinely measured*, keeps the
entire stack Apache/MIT, and runs fully offline. The only cleanly-permissive **real** corpus,
`Sigurdur/icelandic-sign-language` (Apache-2.0, 214 rows, a YouTube-SL-25 slice with `video_id` + timed transcript),
serves as a **real-data smoke test** — not as the training set.

---

## 5. GPU tiers and batch sizes

All training is single-GPU and student-scale. The synthetic spine means even the T4 free tier is sufficient for a full
run; larger tiers only shorten wall-clock. Batch sizes below are for the **t5-small translator** and the **~5M-param pose
transformer recognizer** at the default sequence/segment lengths; they are conservative starting points (raise until
~80–90% VRAM, lower if you hit OOM).

| GPU tier | VRAM | t5-small (gloss→text) batch | pose-transformer recognizer (~5M) batch | Notes |
|---|---|---|---|---|
| **T4** (Colab free) | 16 GB | 16–32 | 64–128 | **Reference tier — everything fits.** fp16; the whole P01 run completes here. |
| **L4** | 24 GB | 32–48 | 128–256 | Good price/perf upgrade; bf16; ~1.5–2× faster than T4. |
| **A100 (40/80 GB)** | 40 / 80 GB | 64–128 / 128–256 | 256–512 / 512–1024 | bf16; overkill for these model sizes — use for fast sweeps / many seeds, not because it is needed. |
| **H100** | 80 GB | 128–256 | 512–1024+ | bf16/fp8; far beyond requirements; only relevant for large hyperparameter grids. |

Guidance:

- **CPU-only / offline path needs no GPU at all.** The `SeedPoseEngine` + numpy nearest-centroid recognizer + lexicon
  translator run on CPU in milliseconds. This is the path the offline tests, the BM-free deployment, and the Docker
  default use.
- **Default to T4.** Given the model sizes (60.5M translator, ~5M recognizer) and the small synthetic vocabulary, the T4
  free tier is the *intended* training environment. Higher tiers buy speed for sweeps, not feasibility.
- **Precision:** fp16 on T4, bf16 on L4/A100/H100. Enable gradient checkpointing only if you scale the translator up to
  m2m100_418M on a 16 GB card (rarely needed).

---

## 6. Honesty caveat on metrics (carried from the design)

Model selection cannot be separated from how the model is judged. The recognizer is scored by **gloss-WER +
position-aligned accuracy + sequence exact-match**; the translator by **BLEU-1..4 (BLEU-4 = headline) + chrF + WER**,
against baselines (most-frequent gloss, random gloss, identity-translate, and the Seed **oracle** upper bound). But
**automatic SLT metrics are unreliable** — length-sensitive and blind to hallucination / semantic equivalence (Yazdani et
al., 2025, hf.co/papers/2510.25434). We therefore report the standard set **and flag this limitation**, and the model
choices lean on this caveat directly: the value of the small, interpretable, abstaining core (per-sign confidence + D5
abstention) is precisely that it does **not** produce a fluent, high-BLEU hallucination from noisy or out-of-vocabulary
signing the way a blind end-to-end decoder would. See `translation_evaluation.md`.

---

## 7. Selection summary

| Decision | Choice | License | Why |
|---|---|---|---|
| Pose front-end (frozen) | **MediaPipe Holistic** / SeedPoseEngine offline | Apache-2.0 | clean license; standard SLT landmark layout; privacy-appropriate; free/offline |
| Frozen-video alternative | `microsoft/xclip-base-patch32` | MIT | permissive escape hatch for video→features mode |
| Front-ends rejected | `MCG-NJU/videomae-base`, `sign/mediapipe-vq` | CC-BY-NC / CC-BY-NC-SA | **non-commercial — AVOID** |
| Recognizer (trained) | compact pose transformer (Colab) / numpy nearest-centroid (offline) | ours | the measured core; ~5M is student-scale (`manohonsy/how2sign-pose-cslr`, MIT, ref) |
| Translator (trained) | **`google-t5/t5-small`** | Apache-2.0 | PRIMARY; 60.5M, fits T4, reuses P13/P14 MT harness |
| Translator alt (multilingual) | `facebook/m2m100_418M` | MIT | reuse P13/P14; non-English targets |
| Translator alt (byte-level) | `google/byt5-small` | Apache-2.0 | OOV-robust / symbolic source |
| Pretrained SLT checkpoint | **none** | — | none load cleanly as Sign→Text; corpora NC/gated → **train our own** |

Bottom line: **MediaPipe Holistic (Apache) for the frozen pose front-end, a ~5M-param transformer / numpy-centroid
recognizer as the trained pose→gloss core, and t5-small (Apache) as the trained gloss→text translator — every
load-bearing component Apache-2.0 or MIT, fully runnable offline on CPU and trainable on a single T4, with no
non-commercial or gated dependency anywhere in the path.**
