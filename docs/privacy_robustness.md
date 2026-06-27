# P01 — Sign Language Translation · Privacy & Robustness

**Author:** Le Dinh Minh Quan (23127460) · **Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`

This document covers the two operational-risk pillars for P01: **privacy** (because sign-language video is
biometric, identifying, Deaf-community data) and **robustness** (because the input is noisy continuous motion under
real-world capture conditions). Both are concrete to this project's pipeline:

```
VIDEO ──(MediaPipe Holistic, frozen)──► pose-keypoint sequence ──► segment ──► recognize gloss ──► translate ──► assemble
```

and to the deterministic 5-decision agent (`src/signlang/agent/`): **D1** ingest/frame-gate → **D2** motion-segment →
**D3** per-segment gloss + confidence → **D4** translate + verify → **D5** abstain-or-answer. Wherever a privacy or
robustness control maps onto a gate, that mapping is called out explicitly.

---

## 1. Privacy

### 1.1 Why this task is unusually privacy-sensitive

Unlike P01's 18 predecessor projects (which consume text or document images), the input here is **video of a person's
face and hands**. That makes the raw input a triple liability:

1. **Biometric.** Hand geometry, gait of motion, and especially the **face** captured by MediaPipe Holistic
   (hand + body + **face** landmarks) are biometric identifiers. Under GDPR Art. 9 and comparable regimes, biometric
   data used to identify a person is a *special category* requiring heightened protection. Even the **derived
   pose-keypoint sequence** (the `2×21` hand + `25` body × 3-coord vectors per frame in `pose/layout.py`, plus the
   face landmarks MediaPipe Holistic returns) is identifying — pose/gait is a known soft biometric, so "we only keep
   keypoints, not pixels" is **not** an adequate anonymization claim.
2. **Identifying content.** Sign language is articulated *on and around the face*. Facial landmarks, mouthing, and
   eye-gaze are linguistically required channels — you cannot strip the face without destroying the signal. The
   utterance content itself (medical, legal, personal) is often sensitive.
3. **Community data.** Sign-language corpora are **Deaf-community** data. The field has a documented history of data
   being collected, modeled, and deployed *about* Deaf signers without their governance. Privacy here is inseparable
   from the ethics of consent and representation (see `ethics_statement.md`).

### 1.2 Default posture: process locally, retain nothing

The product default is the most protective configuration; anything less is opt-in and logged.

- **On-device / edge processing by default.** MediaPipe Holistic is an algorithmic Google library (Apache-2.0) that
  runs **client-side / on the edge**. The reference deployment is designed so pose extraction can happen on the user's
  device, and only the **derived keypoint sequence** (never raw video) need cross a network boundary. The fully offline
  path — `SeedPoseEngine` + numpy nearest-centroid recognizer + lexicon translator — runs entirely on CPU with **no
  network, no MediaPipe, no torch**, which is the strongest privacy guarantee: nothing leaves the machine at all.
- **No retention by default.** Neither raw video nor extracted pose sequences are persisted by default. The FastAPI
  `POST /translate` handler treats inputs as **ephemeral**: decode → translate → return → drop. There is no
  store-by-default, no implicit training-data collection, and no request body logging of pose/video payloads. Any
  retention (e.g. for a user who explicitly opts into improving the model) must be an explicit, separately-consented,
  time-boxed flag — never the default.
- **The optional LLM brain is OFF by default.** The agent's advisory `anthropic` brain (a "please repeat / low
  confidence" note generator) is disabled by default and, critically, **never sees raw video or face landmarks** — it
  is advisory-only and never changes the output. Keeping it off by default means **no biometric-derived data is sent
  to any third-party API** in the default configuration. If a deployer turns it on, it should receive only
  already-abstract symbols (glosses / confidence scores), never pose coordinates or pixels.
- **Encryption in transit and at rest.** When a network boundary is unavoidable (server-side pose extraction, a hosted
  HF Space, or a deployer who enables opt-in retention), pose/video payloads must travel over TLS and any opt-in
  retained artifact must be encrypted at rest with a short, declared lifetime and a deletion path.

### 1.3 Consent and data subject rights

- **Consent is a precondition, not a checkbox.** Capturing a person signing is capturing their face and hands;
  informed consent (purpose, retention, sharing, opt-out) must be obtained before capture, and re-confirmed if the
  purpose changes. For third-party footage (e.g. the real-data smoke test below), consent travels with the dataset
  license.
- **Right to deletion / no silent reuse.** Because nothing is retained by default, the deletion surface is small —
  but any opt-in retained data must be deletable on request and must not be silently repurposed for training.
- **Children and clinical/legal contexts.** In medical or legal interpreting, the privacy stakes compound with the
  accuracy stakes: a wrong-and-retained translation is doubly harmful. The agent's **abstention** (D5) plus
  human-in-the-loop requirement (see `ethics_statement.md`) is itself a privacy control — it stops the system from
  fabricating and recording a confident-but-wrong sensitive utterance.

### 1.4 Dataset-licensing as a privacy/consent signal

The same restrictive licensing that shapes the model stack is a **consent signal**, and every restrictive source is
flagged rather than quietly used:

- **Permissive, used as the real-data smoke test:** `Sigurdur/icelandic-sign-language` (**Apache-2.0**, 214 rows,
  a YouTube-SL-25 Icelandic slice with `video_id` + timed transcript) and `om192006/sign_language_keypoints`
  (**MIT**, 29 isolated gestures). Even here, note these are **public-figure / YouTube** sources — permissive license
  ≠ individual consent to be modeled, so they are used for **smoke-testing the pipeline shape**, not for claiming
  signer-consented training.
- **FLAGGED non-commercial / gated / unspecified (NOT used as the measured corpus):**
  `Exploration-Lab/iSign` (CC-BY-NC-SA + **GATED**), `aipieces/How2Sign` and `PSewmuthu/How2Sign_Holistic`
  (How2Sign upstream is **CC-BY-NC**), `Voxel51/WLASL` (license: other), `Kibalama/poseformer-sign-language`
  (WLASL-derived). **RWTH-PHOENIX-2014T / CSL-Daily / YouTube-ASL** are academic-license corpora **not redistributable
  as HF repos** — no ids are invented for them.
- **Privacy-clean primary data:** the **synthetic pose-sequence generator** (`data/synth_pose.py`) contains **no real
  human** — trajectories are deterministic strokes in keypoint space with embedded gold gloss/text. The primary
  offline training and evaluation therefore carry **zero biometric exposure**, which is the strongest possible privacy
  story for a reproducible baseline.

---

## 2. Robustness

SLT is hard precisely because the input is **continuous, noisy human motion** captured under uncontrolled conditions.
P01's design treats robustness as a first-class, *measured* property and routes each failure mode to a specific gate
that degrades gracefully — ideally into **abstention** rather than a confident hallucination. The guiding principle,
backed by the project's honesty caveat (Yazdani et al. 2025, `hf.co/papers/2510.25434` — automatic SLT metrics are
length-sensitive and blind to hallucination), is: **a system that abstains beats a blind end-to-end decoder that
produces fluent, wrong text from noise.**

### 2.1 Failure modes and the gates that mitigate them

| Failure mode | Why it happens in SLT | Mitigating gate(s) | Behavior |
|---|---|---|---|
| **Too-short / empty / corrupt input** | dropped frames, truncated clip, wrong modality | **D1** frame-count gate (`min_frames`); video↔pose routing | fail fast / route, never decode garbage |
| **Noisy pose from low-quality video** | low resolution, motion blur, dropped landmarks | **D3** per-segment confidence (`recog_min_conf=0.15`) → **D5** abstain | low-confidence flag → "uncertain" + `needs_review` |
| **Multi-sign segmentation errors** | signs run together; no clean rest frame | **D2** motion-based segmentation (low-velocity rest frames split signs) | over/under-segment surfaces as D3 low-confidence |
| **Out-of-vocabulary signs** | signer uses a gloss outside the 40-gloss lexicon | **D3** low recognition confidence → **D5** ratio gate | abstain when OOV/low-conf ratio > `oov_abstain_ratio=0.5` |
| **Continuous vs isolated mismatch** | model expects co-articulated continuous signing | **D2** segmenter is built for continuous streams | segments a stream into sign units, not just one label |
| **Signer-independence drift** | new signer's style/anatomy differs from training | **D3** confidence + train/test **signer split** evaluation | lower confidence → abstain rather than overclaim |
| **Lighting / occlusion / out-of-frame hands** | hand off-camera, shadow, body occludes face | drops landmark quality → **D3** low confidence → **D5** | degrade to abstain, not to a fabricated gloss |
| **Pure noise / non-signing motion** | camera shake, fidgeting, non-signer in frame | **D3** every segment low-conf → **D5** abstain | verified offline: pure-noise input **ABSTAINS** |

### 2.2 The gates in detail

**D1 — frame-count gate (ingest).** The cheapest robustness check: reject inputs below `min_frames` before any
expensive processing, and route the input by modality (raw video → pose extraction vs. already-pose). This stops the
pipeline from "translating" a one-frame or empty clip and forms the outer boundary of the trust region.

**D2 — motion-based segmentation.** Continuous SLT's defining difficulty is that signs are **co-articulated** — there
are no spaces. P01 segments on **low-velocity rest frames**: the synthetic generator deliberately separates signs by
near-still rest frames (so a velocity-threshold segmenter recovers boundaries), and the same motion-energy logic
generalizes to real pose streams. Segmentation quality is **measured directly** via **boundary-F1 vs. gold sign
boundaries**; offline on clean synthetic data this hits **F1 = 1.0**, giving a known ceiling against which real-data
degradation is quantified. Crucially, a segmentation error does not silently corrupt the output — a mis-split segment
produces an ambiguous mean-displacement, which surfaces downstream as **low D3 confidence**.

**D3 — per-segment gloss recognition + confidence.** Each segment is classified independently (offline: a pure-numpy
nearest-centroid classifier over the segment's mean-pose displacement; on Colab: a compact transformer encoder over
pose frames, in the spirit of `manohonsy/how2sign-pose-cslr`, MIT, ~4.8M params — proof a ~5M pose model is
student-scale). Each gloss carries a **confidence**; any segment below `recog_min_conf=0.15` is flagged
**low-confidence**. This is the primary robustness sensor: noise, occlusion, OOV signs, and signer drift all manifest
as **low per-segment confidence** rather than as a confident wrong gloss. The **robustness sweep** in the evaluation
suite injects increasing pose noise and shows recognition degrading monotonically — confidence tracks input quality,
which is exactly what makes the downstream abstention trustworthy.

**D4 — translate + verify.** The gloss→text translator (offline: lexicon map; on Colab: `t5-small`, Apache-2.0 /
`m2m100_418M`, MIT, reused from P13/P14) is followed by a **verification step**: an optional round-trip
text→gloss agreement check and a **chrF keep-gate**. This catches the case where recognition was confident but the
translation drifted (fluent but unfaithful output), re-flagging it instead of emitting it.

**D5 — abstention (finalize).** The terminal safety net. If the ratio of OOV/low-confidence segments exceeds
`oov_abstain_ratio=0.5`, the agent **abstains**: it returns `"uncertain"` + `needs_review` rather than a guessed
sentence. Otherwise it returns glosses + text + **per-sign confidence**. This is the single most important robustness
property of the system and is **verified offline**: pure-noise input abstains, all 5 decision points fire, and the
abstention rate is reported as a first-class metric alongside BLEU/chrF/WER.

### 2.3 Signer-independence and evaluation discipline

Representation/robustness failure is **acute** in SLT: a model trained on one sign language or one set of signers
fails on others. P01 addresses this on two fronts:

- **Evaluation:** report on **train/test signer splits** (held-out signers, not just held-out utterances) so that
  reported accuracy reflects *signer-independent* performance, and run the **pose-noise robustness sweep** to chart
  graceful degradation. Baselines (**most-frequent gloss** ≈ 0.02 floor, **random gloss**, **identity-translate**
  ≈ BLEU 84 showing the lexicon adds ~15 BLEU, and the **Seed oracle** translate-stage upper bound) bound the trained
  core from below and above so robustness claims are calibrated, not absolute.
- **Behavior:** the system **never presents a low-confidence translation as authoritative**. Per-sign confidence is
  always surfaced (in the API response and the Gradio demo), and abstention + human-in-the-loop are mandatory in
  high-stakes (medical/legal) settings.

### 2.4 Why the cascade helps robustness

The Sign2Gloss2Text **cascade** (vs. a black-box end-to-end decoder) is itself a robustness choice: each stage is
**individually measurable** (segmentation-F1, gloss-WER + position-aligned accuracy + sequence exact-match, then
BLEU/chrF/WER), so a degradation can be **localized** to the segmenter, the recognizer, or the translator. A
monolithic model would hide which stage failed and would happily decode fluent text from noise; the cascade exposes
the failure at the stage where confidence collapses and lets **D5** convert it into an honest abstention.

---

## 3. Honesty caveat (applies to every robustness number above)

Automatic SLT metrics — BLEU/chrF/ROUGE/BLEURT — are **unreliable**: length-sensitive and blind to hallucination and
semantic equivalence (Yazdani et al., 2025, `hf.co/papers/2510.25434`). The verified offline numbers
(gloss accuracy 1.0, BLEU 99+, segmentation-F1 1.0 on **clean synthetic** data) are best read as a
**correctness ceiling for the wiring**, not as field performance. On real, noisy, signer-independent data these
numbers will drop — which is precisely why the robustness story is built on **confidence + abstention + per-sign
transparency + human-in-the-loop**, not on a single headline score. We report the standard metric set **and flag its
limitations** in this document, the autoreport, and `translation_evaluation.md`.
