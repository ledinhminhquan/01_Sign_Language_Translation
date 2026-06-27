# P01 — Sign Language Translation · Slide Deck Outline

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment (the last & hardest project)
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`

A ~12-slide presentation outline. One slide per section, 3–5 sub-bullets each. Speaker notes in italics. The deck tells one
story: sign-language translation is a **video/pose → text sequence** task that the field frames as a **Sign2Gloss2Text
cascade**; the defining constraint is **restrictive data licensing**; so we train a small seq2seq core on a **synthetic
pose spine**, wrap it in a **5-decision agent that segments, gates confidence, verifies, and abstains**, and we stay honest
about **unreliable SLT metrics** and the **ethics of biometric Deaf-community data**.

---

## Slide 1 — Title

- **Sign Language Translation (SLT): video / pose-keypoint sequences → spoken-language text via an intermediate gloss.**
- Author: Le Dinh Minh Quan · student 23127460 · NLP in Industry final assignment (P01 — the last and hardest project).
- One-line framing: the **only video/pose → text *sequence* task** in the portfolio — unlike the prior 18 text/OCR systems.
- Package `signlang`; offline-first, runs with **no MediaPipe / torch / video / network** via a synthetic pose spine.

*Speaker note: P01 closes an 18-project arc (P02–P20). It reuses that machinery but adds a genuinely new modality — pose
trajectories in keypoint space — so introduce it as "the same engineering discipline, a harder input."*

---

## Slide 2 — Problem & Use Cases

- **Accessibility gap:** Deaf and hard-of-hearing signers and hearing non-signers cannot communicate without a human
  interpreter; interpreters are scarce, costly, and unavailable on demand.
- Use cases: live captioning, kiosk/front-desk triage, education, customer support, everyday hearing↔Deaf interaction.
- **Framing principle (non-negotiable):** this is an **assistive aid, NOT a replacement** for qualified human interpreters,
  especially in medical / legal / safety-critical settings.
- Design consequence stated up front: the system must **show per-sign confidence and abstain** rather than present a
  fluent guess as authoritative.

*Speaker note: lead with the human need and the humility — set the expectation that "abstain" is a feature, not a failure,
so the later agent slide lands as a value-add rather than a limitation.*

---

## Slide 3 — The Cascade Pipeline

- **Sign2Gloss2Text cascade** (Camgöz et al.'s now-universal framing): `video → pose → segment → recognize gloss →
  translate → assemble sentence`.
- Stage 1 **pose front-end** (MediaPipe Holistic / SeedPoseEngine) — **frozen / algorithmic, NOT trained**.
- Stage 2 **motion-based segmenter** — low-velocity rest frames split a continuous stream into sign units.
- Stage 3 **per-segment gloss recognizer** + Stage 4 **gloss→text translator** — together the **single trainable core**.
- Why a cascade: clean, measurable offline spine that mirrors the P13/P14/P15 cascades; gloss is an inspectable
  intermediate (direct Sign2Text kept as a secondary mode).

*Speaker note: emphasize that only the recognizer + translator are learned; everything else is pretrained or deterministic.
This is what keeps the "measured unit" small and student-scale.*

---

## Slide 4 — Data & the Licensing Reality

- **Headline finding (HF Hub, authenticated):** there is **NO permissive, directly-loadable Sign→Text/Gloss→Text seq2seq
  checkpoint and NO permissive continuous-SLT corpus** — every continuous benchmark is non-commercial, gated, or unspecified.
- **Flagged real corpora:** `Exploration-Lab/iSign` (**CC-BY-NC-SA + GATED**; defines SignPose2Text = this exact task),
  `PSewmuthu/How2Sign_Holistic` & `aipieces/How2Sign` (**How2Sign CC-BY-NC upstream**), `Voxel51/WLASL` &
  `Kibalama/poseformer-sign-language` (**license:other / unset**). RWTH-PHOENIX-2014T, CSL-Daily, YouTube-ASL live on
  academic servers — **not redistributable HF repos (no invented ids)**.
- **Permissive but tiny/isolated:** `Sigurdur/icelandic-sign-language` (**Apache-2.0**, 214 rows → real-data **smoke test**),
  `om192006/sign_language_keypoints` (**MIT**, 29 gestures → pose-schema template).
- **Primary offline data = a SYNTHETIC pose-sequence generator** (`data/synth_pose.py`): 40 glosses (`data/lexicon.py`),
  each a deterministic motion direction; a sign = a triangle stroke; signs separated by near-still rest frames; a sentence
  concatenates 2–6 signs; spoken text deliberately ≠ gloss tokens (`THANK-YOU`→"thank you", `ME`→"i").
- Gold `{glosses, text, boundaries}` is **embedded on the sequence** → the `SeedPoseEngine`/`SeedRecognizer` read it back.

*Speaker note: the licensing audit is the intellectual backbone of the project — restrictive data isn't a footnote, it's
why the synthetic spine is the honest, reproducible design choice rather than a shortcut.*

---

## Slide 5 — Pose Front-End, Recognizer & Translator

- **Front-end (pretrained/algorithmic, NOT trained):** MediaPipe Holistic (Apache, Google) extracts **2×21 hand + 25 body
  landmarks × 3 coords** per frame on Colab; `SeedPoseEngine` offline. Permissive frozen-video alternative
  `microsoft/xclip-base-patch32` (MIT). **Avoid** `MCG-NJU/videomae-base` (CC-BY-NC), `sign/mediapipe-vq` (CC-BY-NC-SA).
- Keypoint layout (`pose/layout.py`) mirrors `PSewmuthu/How2Sign_Holistic` so offline → real data is drop-in.
- **Trainable recognizer:** compact transformer encoder over pose frames on Colab; offline a **pure-numpy nearest-centroid
  classifier** of the sign's mean-pose displacement (no torch). Reference `manohonsy/how2sign-pose-cslr` (**MIT, 4.8M**)
  proves a ~5M pose model is student-scale.
- **Trainable translator (gloss→text):** **`google-t5/t5-small`** (**Apache, 60.5M, DEFAULT**); alternatives
  `facebook/m2m100_418M` (MIT, reuse P13/P14), `google/byt5-small` (Apache, byte-level for symbolic sources).
- **Precedent:** `sign/sockeye-signwriting-to-text` (MIT) — "treat the recognized symbolic sequence as a source language,
  run standard MT." No pretrained Sign→Text loads cleanly into `transformers` → **the trained core is our own small seq2seq.**

*Speaker note: every id on this slide was verified on the Hub; call out that we flag licenses inline so a reviewer can
trust the stack is genuinely deployable.*

---

## Slide 6 — Metrics & the Reliability Caveat

- **Recognition (CSLR stage):** **gloss-WER** (sub/del/ins) + position-aligned gloss accuracy + sequence exact-match.
- **Translation:** **BLEU-1..4 (BLEU-4 = headline)** + chrF/chrF++ + ROUGE-L + WER (reuse P13/P14 MT metrics).
- **Segmentation:** boundary-F1 vs gold sign boundaries. Plus **abstention rate**.
- **Honesty caveat (cited):** Yazdani et al. 2025 (`hf.co/papers/2510.25434`) — automatic SLT metrics
  (BLEU/chrF/ROUGE/BLEURT) are **unreliable**: length-sensitive, blind to hallucination and semantic equivalence. We
  report the standard set **and flag this** in docs + autoreport.
- **Baselines that isolate the core:** identity/passthrough (gloss tokens as text), most-frequent gloss, random gloss,
  **Seed ORACLE** (perfect recognition → translate-stage upper bound).

*Speaker note: the caveat is the credibility slide — we report BLEU-4 because the field does, but we explicitly say it
cannot detect a confident hallucination, which is exactly why abstention matters.*

---

## Slide 7 — The 5-Decision Agent (the value-add)

- **Deterministic FSM, 5 decision points, every step traced** (`src/signlang/agent/`):
- **D1 ingest** — frame-count gate (`min_frames`); route video → pose vs already-pose.
- **D2 segment** — motion-based: low-velocity rest frames split signs. **D3 recognize** — per-segment gloss + confidence;
  below `recog_min_conf = 0.15` → low-confidence.
- **D4 translate+verify** — gloss→text + a round-trip text→gloss agreement / chrF keep-gate. **D5 finalize** — **ABSTAIN**
  if low-confidence-segment ratio > `oov_abstain_ratio = 0.5` → "uncertain" + `needs_review`; else glosses + text +
  per-sign confidence.
- **Value-add:** sign segmentation + per-sign confidence gating + translation verification + **abstention** — beats a blind
  end-to-end decode that hallucinates fluent text from noise.
- Optional LLM brain (`anthropic`) **OFF by default**, advisory only (a "please repeat" note), **never changes output**.

*Speaker note: walk D1→D5 left to right; the punchline is D5 — abstaining on pure noise is the behavior automatic metrics
can't reward but users desperately need.*

---

## Slide 8 — Results

- **Clean synthetic:** gloss accuracy **1.0**, BLEU **99+**, segmentation-F1 **1.0** — vs most-frequent-gloss floor **0.02**.
- **Lexicon earns its keep:** identity-translate (gloss tokens as text) BLEU **~84** → the learned lexicon/reordering adds
  **~15 BLEU**.
- **Robustness sweep:** injected pose noise **degrades recognition** monotonically (signer-independence / low-quality-video
  proxy); the trained core stays above the most-frequent / random baselines.
- **Abstention works:** **pure-noise input ABSTAINS** instead of emitting fluent garbage; the Seed oracle bounds the
  translate stage.
- **All 5 agent decision points fire** on the held-out synthetic split — pipeline verified end-to-end offline.

*Speaker note: the contrast 1.0 vs 0.02 vs identity-84 is the money chart — it shows recognition, the lexicon, and the
oracle gap as three separable contributions.*

---

## Slide 9 — Deployment

- **FastAPI `POST /translate`:** `seed | frames` → glosses + spoken text + **per-sign confidence + abstain flag**.
- **Gradio demo** for interactive pose/video translation; **HF Space** for sharing.
- **Docker** image bundles `mediapipe + ffmpeg + libGL` for the real pose front-end.
- **Offline BM-free path:** SeedPoseEngine + numpy nearest-centroid + lexicon runs on **CPU, no network** — the same code
  path that powers the tests.
- Edge/on-device posture chosen deliberately (see ethics) — keep biometric video local.

*Speaker note: stress that the abstain flag is a first-class field in the API response, not a log line — downstream UIs are
expected to surface "uncertain / needs review."*

---

## Slide 10 — Ethics, Privacy & Representation Bias

- **Biometric + identifying data:** sign-language video captures **face and hands** — Deaf-community data → require consent,
  on-device/edge processing, **no retention by default**, LLM brain OFF.
- **Representation bias is acute:** a model trained on one sign language / one signer set **fails on others** → never
  present a low-confidence translation as authoritative.
- **Mitigations in the system:** abstain + per-sign confidence + human-in-the-loop, especially medical/legal.
- **Assistive, not a replacement** for human interpreters; **engage the Deaf community** in design and evaluation.
- Robustness frontier acknowledged: noisy pose, continuous-vs-isolated signs, signer-independence, OOV signs, low-resource /
  restrictive licensing.

*Speaker note: tie privacy and bias back to concrete code decisions (retention off, abstain ratio, confidence in the API)
so ethics reads as engineered, not aspirational.*

---

## Slide 11 — Reuse & Engineering

- **From P13/P14 (MT):** BLEU/chrF/WER implementations + the seq2seq train/eval pattern + t5/m2m100 backbone.
- **From P15/P17/P19/P20:** the embedded-gold synthetic generator + Seed/Stub offline pattern; standard
  config/logging/registry/autoreport/charts/monitoring/automation/grading/cli/api templates.
- **NEW for P01:** pose-keypoint layout + front-end (MediaPipe / SeedPoseEngine), the synthetic **pose-sequence** generator
  (trajectories, not images), the motion-based **segmenter**, the per-segment **gloss recognizer** (numpy centroid offline
  + transformer on Colab), the sign-translation agent.
- One coherent pattern across 19 systems: deterministic offline spine → optional pretrained upgrade on Colab.

*Speaker note: this slide answers "is this just glued-together libraries?" — no; the shared scaffolding is reused, the
modality-specific core (pose, segmentation, recognition) is genuinely new.*

---

## Slide 12 — Conclusion & Future Work

- **Conclusion:** a complete, honest, offline-reproducible SLT cascade — pose → segment → recognize → translate — with a
  5-decision agent whose headline behavior is **knowing when to abstain**.
- **Future work — modeling:** **gloss-free Sign2Text** (skip the gloss bottleneck), larger pose transformers, joint
  segmentation+recognition (CTC).
- **Future work — data:** train on **real corpora with proper consent** (How2Sign / iSign under their licenses; PHOENIX /
  CSL-Daily via academic agreements) and validate on the Icelandic smoke test.
- **Future work — coverage:** **more sign languages and signers** to attack representation bias; signer-independent eval.
- **Future work — evaluation:** semantic / human-in-the-loop metrics beyond BLEU/chrF, given the documented unreliability.

*Speaker note: close on the throughline — the hardest project in the portfolio, solved honestly: real constraints
(licensing, biometrics, unreliable metrics) faced head-on rather than papered over.*
