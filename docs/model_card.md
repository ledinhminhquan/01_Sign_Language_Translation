# Model Card — P01 Sign Language Translation (Sign2Gloss2Text core)

**Author:** Le Dinh Minh Quan (student 23127460) · NLP in Industry, Final Assignment.
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`.
**Model name:** `signlang-sign2gloss2text` (the trained sign-to-gloss/text core).
**Card version:** matches the repo release; last reviewed 2026-06-27.

> This card documents only the **trainable core** of the P01 cascade — the per-segment
> **pose → gloss recognizer** and the **gloss → text translator**. The pose front-end
> (MediaPipe Holistic / `SeedPoseEngine`) and the motion-based segmenter are **not trained
> models** and are described here only as context. See `docs/architecture.md`,
> `docs/agent_architecture.md`, and `docs/translation_evaluation.md` for the full pipeline,
> the deterministic agent, and the evaluation methodology.

---

## 1. Model overview

P01 maps a sign-language **video / pose-keypoint sequence → spoken-language text** through an
intermediate **gloss** sequence (the now-standard Sign2Gloss2Text cascade, after Camgöz et al.).
This is a **video/pose → text sequence** task — fundamentally different from the prior 18
text/OCR projects in this assignment series, which is why P01 is the last and hardest.

The cascade is:

```
VIDEO ──(MediaPipe Holistic, frozen / algorithmic — NOT trained)──► pose-keypoint sequence
      ──► motion-based SEGMENT into sign units
      ──► RECOGNIZE gloss per segment        ◄── TRAINABLE core, stage 1
      ──► TRANSLATE gloss → spoken text       ◄── TRAINABLE core, stage 2
      ──► assemble sentence
```

**Only the sign-to-gloss/text seq2seq is trained.** The pose extractor is pretrained /
algorithmic and frozen; the segmenter is a deterministic motion heuristic. The two trained
stages are the single measured unit this card describes.

### 1.1 The two trainable stages

| Stage | Offline (default, CPU, no torch) | Colab / GPU upgrade | Role |
|---|---|---|---|
| **Stage 1 — pose → gloss recognizer** | pure-**numpy nearest-centroid** classifier over per-segment mean-pose displacement | compact **transformer encoder** over pose frames (~5M params, student-scale) | classify each motion segment into one of 40 glosses + a confidence |
| **Stage 2 — gloss → text translator** | **lexicon** map (gloss → text) | **`google-t5/t5-small`** (Apache-2.0, 60.5M) seq2seq, fine-tuned gloss→text | reorder + lexicalize the gloss string into fluent spoken text |

Both stages are deliberately small. The offline path runs the **entire** pipeline
(segment → recognize → translate → eval → agent → tests) with **no MediaPipe, no torch, no
video, no network** — the numpy centroid recognizer genuinely classifies the sign's
mean-pose displacement rather than reading a label, so the offline metrics are real, not faked.

### 1.2 Architecture references (not loaded as checkpoints)

- **`manohonsy/how2sign-pose-cslr`** (MIT, 4.8M params, pose + CTC CSLR on How2Sign) — the
  proof point that a **~5M-parameter pose model is student-scale** and sufficient for the
  recognition stage. Used as an *architecture reference only*; it is not loaded as our measured
  component.
- **`sign/sockeye-signwriting-to-text`** (MIT) — the precedent for the central design move:
  *treat the recognized symbolic (gloss) sequence as a source language and run standard MT*.
  Built on Sockeye, **not** loadable into `transformers`; used as a design precedent only.

> **No pretrained Sign→Text/Gloss→Text checkpoint loads cleanly into `transformers`.** There is
> no permissively-licensed, directly-loadable SLT seq2seq on the Hub. The trained core is
> therefore **our own small seq2seq**, built on a permissive backbone — this constraint is
> structural to the field and is the reason for the synthetic-spine design.

---

## 2. Intended use

### 2.1 Primary intended use
- **Assistive sign → text drafting.** Convert a segment of signing (as pose keypoints or video)
  into a *draft* gloss sequence and a *draft* spoken-text translation, **with per-sign
  confidence and an explicit abstain flag**.
- Education / research demonstration of a Sign2Gloss2Text cascade with honest, reproducible
  offline evaluation under restrictive-licensing constraints.

### 2.2 Intended users
- Accessibility-tool developers building **human-in-the-loop** drafting aids.
- NLP students / researchers studying SLT cascades, gloss recognition, and SLT evaluation.

### 2.3 Out-of-scope and prohibited uses
- **Not an interpreter substitute.** This model must **not** be used as the sole channel in any
  setting where a mistranslation causes harm — **medical, legal, emergency, employment, or
  financial** contexts require a qualified human interpreter.
- **Not authoritative.** A produced translation is a draft. A low-confidence or abstained output
  must never be presented as a confirmed translation.
- **Not validated for production sign languages.** The trained core is fitted on a **synthetic
  40-gloss vocabulary** (plus a real-data smoke test); it does not cover any natural sign
  language's full lexicon, grammar, or signer population.
- No surveillance, identification, or profiling of signers. Sign-language video is biometric
  data (see §6).

---

## 3. Inputs and outputs

**Input.** A pose-keypoint sequence with the layout in `pose/layout.py`:
**2×21 hand + 25 body landmarks × 3 coords** per frame — the same per-frame shape MediaPipe
Holistic and `PSewmuthu/How2Sign_Holistic` produce. Video input is converted to this layout by
the frozen front-end (MediaPipe Holistic on Colab; `SeedPoseEngine` offline). The API also
accepts a `seed` spec for the offline path.

**Output (per request).**
- `glosses` — recognized gloss sequence (one gloss per segment).
- `text` — assembled spoken-language translation.
- `per_sign_confidence` — a confidence value per recognized segment.
- `abstain` / `needs_review` — set when the model is uncertain (see §4 and the agent doc).
- segment boundaries (from the motion segmenter).

The gloss vocabulary is deliberately **not** the text vocabulary — e.g. `THANK-YOU` → "thank
you", `ME` → "i" — so the translation stage performs genuine reordering and lexicalization, not
an identity copy.

---

## 4. Agent behavior (confidence gating + abstention)

The trained core runs inside a deterministic 5-decision FSM (`src/signlang/agent/`):

1. **D1 ingest** — frame-count gate (`min_frames`); route video vs already-pose.
2. **D2 segment** — motion-based segmentation; low-velocity rest frames split signs.
3. **D3 recognize** — per-segment gloss + confidence; below `recog_min_conf = 0.15` → flag
   low-confidence.
4. **D4 translate + verify** — gloss → text, then a round-trip text→gloss agreement /
   **chrF keep-gate**.
5. **D5 finalize** — **ABSTAIN** if the low-confidence-segment ratio exceeds
   `oov_abstain_ratio = 0.5` → returns `"uncertain"` + `needs_review`; otherwise returns
   glosses + text + per-sign confidence.

The optional LLM brain (`anthropic`) is **OFF by default**, advisory only (e.g. a "please
repeat" note), and **never changes the output**. The value-add of the agent is exactly this:
**segmentation + per-sign confidence gating + translation verification + abstention** — it
beats a blind end-to-end decode that hallucinates fluent text from noisy or out-of-vocabulary
signing.

---

## 5. Training data

### 5.1 Primary — synthetic pose-sequence generator (`data/synth_pose.py`)
The primary training and evaluation data is a **deterministic synthetic pose-sequence
generator**, mirroring the embedded-gold pattern of P15/P17/P19/P20:

- A small vocabulary of **40 ASL-style glosses** (`data/lexicon.py`), each with a gloss→text map.
- For each gloss, a deterministic **motion direction** in keypoint space (seeded by gloss index).
- A **sign** = a triangle stroke along that direction, so the per-sign **mean displacement
  recovers the gloss**.
- Signs are separated by near-still **rest frames**, so the motion-based segmenter can split them.
- A **sentence** concatenates **2–6 signs**; the spoken text comes from the lexicon and is
  deliberately **≠** the gloss tokens.
- The gold spec `{glosses, text, boundaries}` is **embedded on the sequence**, so the
  `SeedPoseEngine` / `SeedRecognizer` read it back for oracle baselines and tests.

This is a **defensible, honest** design choice, not a shortcut: restrictive licensing is *the*
defining constraint of continuous SLT, so a reproducible synthetic generator that embeds gold
gloss/text is the appropriate primary data. Evaluation uses a **held-out synthetic split**.

### 5.2 Real-data smoke test (permissive)
- **`Sigurdur/icelandic-sign-language`** — **Apache-2.0** ✅, 214 rows, a YouTube-SL-25 Icelandic
  slice (`video_id` + timed `transcript`). The **only cleanly permissive real corpus** on the
  Hub; used as a real-data **smoke test**, not as the primary training set.
- **`om192006/sign_language_keypoints`** — **MIT** ✅, pre-extracted MediaPipe keypoints, 29
  isolated gestures; tiny/isolated, used as a pose-schema template.

### 5.3 Flagged real corpora (NOT used for training — license-restricted)
The following were reviewed during research and are **not** used as training data because of
their licenses. They are listed for transparency and to warn downstream users.

| id | License | Flag |
|---|---|---|
| `Exploration-Lab/iSign` | **CC-BY-NC-SA + GATED** | non-commercial + ShareAlike + gated; defines `SignPose2Text` (this exact task) |
| `aipieces/How2Sign` | unspecified | upstream How2Sign is **CC-BY-NC** |
| `PSewmuthu/How2Sign_Holistic` | MIT tag | **derived from How2Sign (NC upstream)** — treat as NC |
| `Voxel51/WLASL` | `other` | research terms; isolated signs; recognition only |
| `Kibalama/poseformer-sign-language` | unset (WLASL "other") | isolated, non-permissive source |

**Not redistributable on the Hub (do not invent ids):** RWTH-PHOENIX-Weather-2014T, CSL-Daily,
YouTube-ASL / YouTube-SL-25 — academic licenses on university servers. The **avoided
non-commercial / unspecified models** include `MCG-NJU/videomae-base` (CC-BY-NC),
`sign/mediapipe-vq` (CC-BY-NC-SA), and `sign/sockeye-text-to-factored-signwriting` (CC-BY-NC).

### 5.4 Backbone provenance and licenses (translator)
- **`google-t5/t5-small`** — **Apache-2.0**, 60.5M params. DEFAULT translator backbone.
- `google/byt5-small` — Apache-2.0, byte-level; robust to OOV glosses / symbolic pose tokens.
- `facebook/m2m100_418M` — MIT; reuses the P13/P14 MT harness for multilingual gloss→text.

All three backbones are **permissively licensed** (Apache-2.0 / MIT). The pose front-end
(MediaPipe Holistic, Apache-2.0, Google) is algorithmic and not retrained.

---

## 6. Evaluation and metrics

Metrics are reused from P13/P14 (MT) plus recognition/segmentation metrics new to P01. Reported
on the held-out synthetic split (and as a smoke test on the Icelandic slice).

**Recognition (CSLR stage):**
- **gloss WER** (substitutions/deletions/insertions over gloss tokens),
- position-aligned **gloss accuracy**,
- **sequence exact-match**.

**Translation:**
- **BLEU-1..4** (**BLEU-4 = headline**), **chrF / chrF++**, ROUGE-L, plus WER.

**Segmentation:** **boundary-F1** vs the gold sign boundaries.

**Abstention:** abstention rate is reported alongside accuracy (an abstain is not a wrong answer).

**Baselines (to isolate the trained core):**
- **most-frequent gloss** recognizer,
- **random gloss**,
- **identity / passthrough translate** (gloss tokens used directly as text — shows the
  reordering/lexicon the model actually learns),
- **Seed ORACLE** (perfect recognition → upper bound on the translate stage).
The trained core must beat the first three and approach the oracle.

### 6.1 Verified offline behavior
On clean synthetic data the offline core verifies:

| Quantity | Result |
|---|---|
| gloss accuracy (clean synthetic) | **1.0** |
| translation BLEU (clean synthetic) | **99+** |
| segmentation boundary-F1 | **1.0** |
| most-frequent-gloss floor | **~0.02** |
| identity-translate BLEU | **~84** (the lexicon adds **~15 BLEU** over passthrough) |
| pose-noise robustness sweep | recognition **degrades gracefully** as noise rises |
| pure-noise input | the agent **ABSTAINS** |
| agent decision points | **all 5 fire** |

The gap between the floor (~0.02) and the trained core (1.0), and the ~15 BLEU the lexicon adds
over identity-translate, show the core learns genuine recognition and translation — not a copy.

### 6.2 Evaluation honesty caveat (mandatory)
Automatic SLT metrics — **BLEU / chrF / ROUGE / BLEURT** — are **unreliable**: length-sensitive
and blind to hallucination and semantic equivalence (Yazdani et al. 2025,
`hf.co/papers/2510.25434`). We **report the standard set AND flag this limitation** here, in
`docs/translation_evaluation.md`, and in the autoreport. A high BLEU on synthetic data does
**not** imply real-world translation quality; metrics are a development signal, not a
certification of correctness.

---

## 7. Limitations

- **Synthetic-domain gap.** The headline numbers (gloss accuracy 1.0, BLEU 99+) are on a
  **synthetic** distribution with deterministic, clean trajectories. They are an upper bound on
  pipeline health, **not** a forecast of accuracy on real signing. Natural signing has
  coarticulation, non-manual features (facial grammar), spatial referencing, and signer variation
  the synthetic generator does not model.
- **Vocabulary coverage.** The trained core covers **40 synthetic glosses**. Real sign languages
  have thousands of signs plus productive/classifier constructions — far beyond this vocabulary.
- **Single sign language / signer set.** A model fit on one sign language or signer population
  **fails on others** (representation bias, §8). The synthetic spine encodes one stylized motion
  scheme; there is no validation across real sign languages, dialects, or signers.
- **Out-of-vocabulary (OOV) signs.** Unknown signs are recognized with low confidence and, in
  aggregate, trigger **abstention** rather than a confident wrong answer — but individual OOV
  signs can still be misrecognized as a near-neighbor gloss.
- **Continuous vs isolated.** Segmentation relies on **rest frames** between signs; real
  continuous signing has minimal pauses and heavy coarticulation, which degrades the motion-based
  segmenter and, downstream, recognition.
- **Noisy pose.** Low-quality video, occlusion, motion blur, and lighting degrade MediaPipe
  landmarks; the robustness sweep shows recognition falls as pose noise rises.
- **Metric unreliability.** See §6.2 — good automatic scores do not guarantee faithful
  translations.
- **No pretrained SLT checkpoint.** The core is small and trained by us; it does not benefit from
  large-scale SLT pretraining, because no permissive, loadable SLT checkpoint exists.

---

## 8. Ethical considerations

- **Biometric, identifying data.** Sign-language video captures the **face and hands** and is
  **biometric and personally identifying** Deaf-community data. The system defaults to
  **consent + on-device / edge processing + no retention**, and the LLM brain is **OFF by
  default**. Pose keypoints should be preferred over raw video wherever possible, and inputs
  should not be stored or used to identify signers.
- **Not a replacement for human interpreters.** This is an **assistive aid**. It must not replace
  a qualified interpreter, especially in **medical, legal, and emergency** settings. Keep a human
  in the loop and engage the **Deaf community** in any deployment.
- **Representation bias is acute.** Coverage skews to whatever sign language / signer set the
  model saw (here: a synthetic ASL-style scheme). Performance on other sign languages, regional
  dialects, signer demographics, and signing styles is **untested and expected to be poor**. The
  product mitigations are structural: **never present a low-confidence translation as
  authoritative**, always **show per-sign confidence**, and **abstain** when uncertain.
- **Abstention as a safety feature.** Returning `"uncertain"` + `needs_review` on low-confidence
  or out-of-vocabulary input is preferable to emitting fluent, confident, wrong text — a blind
  end-to-end decoder would hallucinate. Downstream UIs must surface the abstain flag and
  confidences, not hide them.

---

## 9. Deployment

- **Serving:** FastAPI `POST /translate` (`seed` or `frames` → glosses + spoken text + per-sign
  confidence + abstain flag), a Gradio demo, Docker (mediapipe + ffmpeg + libGL), and an HF Space.
- **Offline / CPU path:** `SeedPoseEngine` + numpy nearest-centroid recognizer + lexicon
  translator — no GPU, no torch, no network. This is the default path used for tests, grading,
  and the verified offline behavior in §6.1.
- **Upgrade path:** on Colab/T4 the recognizer becomes a compact transformer encoder and the
  translator becomes a fine-tuned `t5-small` (or `m2m100_418M` via the P13/P14 harness).

---

## 10. Reproducibility and provenance

- **Reused:** P13/P14 MT metrics (BLEU/chrF/WER) + seq2seq train/eval pattern + t5/m2m100
  backbone; the embedded-gold synthetic-generator + Seed/Stub offline pattern from
  P15/P17/P19/P20; the standard config/logging/registry/autoreport/charts/monitoring/automation/
  grading/cli/api templates.
- **New for P01:** the pose-keypoint layout + front-end (MediaPipe / `SeedPoseEngine`), the
  synthetic **pose-sequence** generator (trajectories, not images), the motion-based
  **segmenter**, the per-segment **gloss recognizer** (numpy centroid offline + transformer on
  Colab), and the sign-translation agent.
- **Determinism:** the synthetic generator is seeded by gloss index; the offline pipeline is
  fully deterministic and reproducible without external data or network access.

---

## 11. Summary

`signlang-sign2gloss2text` is a deliberately small, honestly-evaluated Sign2Gloss2Text core: a
numpy/transformer **pose→gloss recognizer** plus a lexicon/`t5-small` **gloss→text translator**,
wrapped in a deterministic agent that **segments, gates on per-sign confidence, verifies the
translation, and abstains** when uncertain. It is trained primarily on a reproducible
**synthetic pose generator** (with a permissive real-data smoke test) because no permissive,
loadable SLT checkpoint or continuous corpus exists. It is an **assistive drafting aid** — built
around biometric-data care, representation-bias awareness, and abstention — and is explicitly
**not** a substitute for a human sign-language interpreter.
