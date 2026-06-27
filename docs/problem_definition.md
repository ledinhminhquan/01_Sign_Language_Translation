# P01 — Sign Language Translation · Problem Definition

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`

This is the last and hardest project in the assignment. Unlike the prior eighteen systems (P02–P15, P17–P20), which all
operate on text or OCR'd text, **P01 takes a visual–temporal input — sign-language video or a pose-keypoint sequence —
and produces spoken-language text**. It is a video/pose → text *sequence* task, and the only project in the set whose
primary signal is human motion rather than characters on a page.

---

## 1. What sign-language translation is

Sign Language Translation (SLT) is the task of converting an utterance produced in a **signed language** (e.g. American
Sign Language, Indian Sign Language) into an equivalent sentence in a **spoken/written language** (e.g. English). The
input is a continuous stream of articulated motion — hands, arms, body posture, and face — captured as **video** or, after
a pose front-end, as a **sequence of keypoint coordinates per frame**. The output is fluent text in the target language.

Crucially, signed languages are **not** word-for-word encodings of spoken languages. They have their own grammar,
spatial syntax, and morphology; sign order frequently differs from spoken-word order; many spoken function words have no
signed counterpart, and many signs map to multiple spoken words. SLT is therefore a genuine **translation** problem, not a
transcription or substitution problem.

### The Sign2Gloss2Text framing

Following the now-standard framing (Camgöz et al.), P01 decomposes SLT into a **cascade** with an intermediate symbolic
representation called a **gloss**. A gloss is a written label for a single sign (conventionally upper-case, e.g.
`THANK-YOU`, `ME`, `BOOK`). The gloss sequence is a compact, language-near transcript of *what was signed*, before it is
rendered into natural spoken text.

```
VIDEO ──(MediaPipe Holistic, frozen / algorithmic — NOT trained)──► pose-keypoint sequence
      ──► motion-based SEGMENT into sign units
      ──► RECOGNIZE one gloss per segment   (Sign2Gloss — the trainable recognition stage)
      ──► TRANSLATE the gloss sequence → spoken text   (Gloss2Text — the trainable translation stage)
      ──► assemble the sentence
```

This **Sign2Gloss2Text cascade** is chosen deliberately:

- It gives a **clean, measurable spine**: the recognition stage and the translation stage can each be evaluated against
  their own gold (gloss WER for recognition, BLEU/chrF for translation), so failures are attributable.
- It **mirrors the P13/P14/P15 cascades**, letting P01 reuse the existing machine-translation harness for the
  Gloss2Text stage.
- It matches how the field's only permissive precedent operates — `sign/sockeye-signwriting-to-text` (MIT) demonstrates
  the "treat the recognized symbolic sequence as a source language and run standard MT" pattern.

Direct **Sign2Text** (skipping the explicit gloss) is supported as a secondary mode, but the gloss-mediated cascade is
the primary, measured design.

### What is pretrained vs. what is trained

A defining property of P01 is that **only the sign-to-gloss/text seq2seq core is trained**. The pose front-end is
pretrained or purely algorithmic and **frozen**:

- **Pose front-end (NOT trained):** `MediaPipe Holistic` (Apache-2.0, Google) extracts hand + body + face landmarks per
  frame on Colab. Offline, a deterministic **`SeedPoseEngine`** reads the gold embedded in synthetic sequences. A
  permissive frozen-video alternative is `microsoft/xclip-base-patch32` (MIT). *We avoid `MCG-NJU/videomae-base`
  (CC-BY-NC) and `sign/mediapipe-vq` (CC-BY-NC-SA) — both non-commercial.*
- **Trainable core (the single measured unit):** (1) a per-segment **pose→gloss recognizer** — a compact transformer
  encoder over pose frames on Colab, and a pure-numpy nearest-centroid classifier offline (no torch) that genuinely
  classifies the sign's mean-pose displacement; and (2) a **gloss→text translator** — `google-t5/t5-small` (Apache-2.0,
  60.5M, default), with `facebook/m2m100_418M` (MIT, reused from P13/P14) and `google/byt5-small` (Apache-2.0, byte-level
  for symbolic sources) as alternates. The reference `manohonsy/how2sign-pose-cslr` (MIT, 4.8M) proves that a ~5M-parameter
  pose model is student-scale.

There is **no permissively-licensed, directly-loadable Sign→Text or Gloss→Text checkpoint on the Hub** — so the trained
core is necessarily *our own small seq2seq*, not a fine-tune of an existing SLT model.

---

## 2. Why it is hard

SLT is materially harder than the prior text-based projects for several compounding reasons:

1. **Continuous vs. isolated signing.** Easy benchmarks recognise *isolated* signs (one sign, clean start and end —
   e.g. `Voxel51/WLASL`, 11,980 isolated clips). Real signing is **continuous**: signs flow into one another with no
   word boundaries, with co-articulation (the end of one sign blends into the start of the next). P01 must therefore
   **segment** a continuous motion stream into sign units before it can recognise them — the motion-based segmenter is a
   first-class component, not an afterthought.

2. **Signer independence.** Signing style, speed, body proportions, handedness, and camera framing vary enormously
   between people. A model that fits one signer often fails on another. This is a domain-shift problem with no
   text-world analogue.

3. **Low-resource and restrictive licensing — the defining constraint.** Every *continuous* sign-language corpus is
   restrictively licensed, gated, or not redistributable:
   - `Exploration-Lab/iSign` (defines `SignPose2Text`, this exact task, 118K+ pairs) — **CC-BY-NC-SA + GATED** (flag:
     non-commercial + ShareAlike + gated).
   - `aipieces/How2Sign` and `PSewmuthu/How2Sign_Holistic` — **flag: derived from How2Sign, whose upstream is CC-BY-NC.**
   - `Voxel51/WLASL` (license `other`) and `Kibalama/poseformer-sign-language` (WLASL-derived, unset) — **flag: research
     terms, isolated signs only.**
   - `RWTH-PHOENIX-Weather-2014T`, `CSL-Daily`, and `YouTube-ASL/SL-25` are **not redistributable Hub repos** — they live
     on university servers under academic licenses, so their ids must not be invented or depended on.
   The only cleanly permissive real corpora are tiny and isolated: `Sigurdur/icelandic-sign-language` (Apache-2.0, 214
   rows, a YouTube-SL-25 slice with `video_id` + timed transcript) and `om192006/sign_language_keypoints` (MIT, 29
   gestures). These serve as a **real-data smoke test**, not as a training corpus.

4. **No off-the-shelf model to lean on.** Because no permissive Sign→Text checkpoint loads cleanly into `transformers`,
   the system cannot be assembled from a pretrained SLT model. The trainable core must be built and trained from scratch
   on a permissive backbone.

5. **Out-of-vocabulary signs and unintelligible input.** Real deployments encounter signs the model has never seen and
   video too noisy to recognise. A blind end-to-end decoder will **hallucinate fluent text from noise** — a dangerous
   failure mode for an accessibility tool. P01 must instead detect low confidence and **abstain**.

6. **Unreliable evaluation.** Automatic SLT metrics (BLEU, chrF, ROUGE, BLEURT) are **length-sensitive and blind to
   hallucination and semantic equivalence** (Yazdani et al., 2025, hf.co/papers/2510.25434). We report the standard
   metric set **and explicitly flag this limitation** rather than treating a high BLEU as proof of correctness.

To make the full pipeline reproducible under these constraints, the **primary offline data is a synthetic pose-sequence
generator** (`data/synth_pose.py`): for each of 40 glosses (`data/lexicon.py`, with a gloss→text map) it fixes a
deterministic motion direction in keypoint space; a sign is a triangle stroke along that direction (so the mean
displacement recovers the gloss), signs are separated by near-still **rest frames** (so the segmenter can split them), and
a sentence concatenates 2–6 signs. Spoken text comes from the lexicon and is deliberately ≠ the gloss tokens
(`THANK-YOU`→"thank you", `ME`→"i"), so translation is non-trivial. The gold `{glosses, text, boundaries}` is embedded on
each sequence and read back by the `SeedPoseEngine`/`SeedRecognizer`. The keypoint layout (`pose/layout.py`) mirrors real
data: 2×21 hand + 25 body landmarks × 3 coords per frame, matching MediaPipe Holistic / `PSewmuthu/How2Sign_Holistic`.

---

## 3. Inputs and outputs

**Inputs (either form):**

- A **sign-language video** (frames), processed on Colab through MediaPipe Holistic into a pose-keypoint sequence; or
- An **already-extracted pose-keypoint sequence** — a per-frame vector of `2×21 hand + 25 body` landmarks × 3 coords,
  the offline default produced by the `SeedPoseEngine` over synthetic data.

**Outputs:**

- The recognised **gloss sequence** (the intermediate symbolic transcript);
- The translated **spoken-language text** (the headline product);
- A **per-sign confidence** for each recognised segment; and
- An **abstain flag** (`uncertain` + `needs_review`) when the input is too low-confidence to translate responsibly.

The deployed API (`POST /translate`) accepts either `seed` or `frames` input and returns glosses, spoken text, per-sign
confidence, and the abstain flag.

---

## 4. Scope and non-goals

**In scope:**

- Sign **video / pose-keypoint sequence → spoken text**, via the Sign2Gloss2Text cascade (with direct Sign2Text as a
  secondary mode).
- Motion-based **segmentation** of continuous signing into sign units.
- Per-segment **gloss recognition** (the trainable recognition stage) and **gloss→text translation** (the trainable
  translation stage).
- An **agentic pipeline** with confidence gating, translation verification, and **abstention**.
- A reproducible **offline path** (synthetic pose generator + numpy centroid recognizer + lexicon translator) requiring
  no mediapipe, torch, video, or network.

**Explicit non-goals:**

- **NOT text → sign avatar synthesis / sign generation.** P01 translates *from* sign *into* text; it does not animate or
  generate signing. (The non-commercial `sign/sockeye-text-to-factored-signwriting` is out of scope for that reason.)
- **NOT a real-time, signer-independent, open-vocabulary production translator.** P01 is a student-scale, honest system
  built around a synthetic spine and a small permissive vocabulary.
- **NOT a replacement for human interpreters.** P01 is an **assistive** aid. Low-confidence translations are never
  presented as authoritative; a human stays in the loop, especially in medical and legal settings.
- **NOT a depender on any gated / non-commercial corpus or any pretrained SLT checkpoint** as a measured component.

---

## 5. Real-world use cases

SLT systems support Deaf and hard-of-hearing communication, always as an **assistive** layer and never as a substitute
for a qualified human interpreter:

- **Accessibility** — captioning signed content (lectures, broadcasts, social video) into text.
- **Video relay / remote communication** — assisting an interpreter-mediated call when no interpreter is immediately
  available; surfacing a draft transcript for review.
- **Captioning** — generating reviewable text tracks for recorded signed media.
- **Public-service kiosks** — accepting a small, fixed signed vocabulary (greetings, common requests) at a counter, with
  abstention when input falls outside that vocabulary.

In each case the system's job is to **assist**, to surface confidence, and to **defer to a human** when uncertain — not
to make authoritative decisions on the signer's behalf.

---

## 6. Success criteria

Success is measured per cascade stage, plus segmentation and the agent's abstention behaviour, with an honesty caveat on
the metrics themselves.

**Recognition (Sign2Gloss / CSLR stage):**

- **Gloss WER** (substitutions / deletions / insertions over gloss tokens) — primary recognition metric;
- position-aligned **gloss accuracy**;
- **sequence exact-match** rate.

**Translation (Gloss2Text stage), reusing P13/P14 MT metrics:**

- **BLEU-1..4**, with **BLEU-4 as the headline**;
- **chrF / chrF++**;
- **ROUGE-L** and **WER**.

**Segmentation:** boundary-**F1** against the gold sign boundaries.

**Agent:** **abstention rate** on low-confidence / out-of-vocabulary input.

**Baselines (to isolate what the trained core actually learns):**

- **most-frequent gloss** and **random gloss** recognizers (recognition floors);
- **identity / passthrough translate** (gloss tokens emitted verbatim as text — shows the reordering and lexicon the
  translator must learn);
- **Seed oracle** (perfect recognition → upper bound on the translate stage).

The trained core must **beat** the most-frequent / random / identity baselines and **approach** the oracle.

**Verified offline results (clean synthetic):** gloss accuracy **1.0**, BLEU **99+**, segmentation-F1 **1.0** — against a
most-frequent floor of **0.02** and an identity-translate BLEU of **~84** (so the lexicon adds **~15 BLEU**). Pose noise
degrades recognition as expected in the robustness sweep; pure-noise input correctly **abstains**; all five agent decision
points fire.

> **Honesty caveat.** Automatic SLT metrics (BLEU / chrF / ROUGE / BLEURT) are **unreliable** — length-sensitive and
> blind to hallucination and semantic equivalence (Yazdani et al., 2025, hf.co/papers/2510.25434). We report the standard
> metric set **and flag this limitation** in the docs and autoreport; a high BLEU is never treated as proof of a correct,
> faithful translation.

---

## 7. Assignment mapping

| Assignment requirement | How P01 satisfies it |
|---|---|
| A real, non-trivial NLP task | Sign video/pose → spoken text via gloss — a video/pose → text *sequence* task, the hardest in the set |
| A **trained** model core | An own small seq2seq: pose→gloss recognizer (transformer / numpy centroid) + gloss→text translator (`t5-small`, Apache-2.0) — no pretrained SLT checkpoint depended on |
| A mandatory **agentic** component | A deterministic 5-decision FSM (ingest → segment → recognize → translate+verify → finalize) with confidence gating, translation verification, and **abstention** |
| Honest, reproducible evaluation | Stage-wise gloss WER/accuracy + BLEU-4/chrF + segmentation-F1 + abstention, with baselines and oracle, plus an explicit metric-reliability caveat |
| License diligence | Every dataset/model id verified on the Hub; non-commercial / gated / unspecified licenses flagged; a permissive synthetic spine as the primary data |
| Deployment | FastAPI `POST /translate` + Gradio demo + Docker (mediapipe + ffmpeg + libGL) + HF Space; a CPU-only offline path |
| Ethics & privacy | Sign video treated as biometric / identifying Deaf-community data — consent, on-device processing, no retention by default, LLM brain off; representation-bias and assistive-not-replacement stance |

P01 thus closes the assignment by extending the established cascade + synthetic-spine + agent pattern (P13/P14/P15/P17/P19/
P20) into the visual–temporal domain, while honestly confronting the field's restrictive-licensing and metric-reliability
realities.
