# P01 — Sign Language Translation · Continual Learning & Monitoring

**Author:** Le Dinh Minh Quan (23127460) · Package `signlang` · Folder `01_Sign_Language_Translation/`

This document covers how the Sign2Gloss2Text cascade is **monitored in production** and **kept current over
time**. Unlike the prior 18 text/OCR projects, P01 ingests a **video / pose-keypoint sequence** and emits a
spoken-text sentence through a multi-stage pipeline (pose front-end → motion segmenter → per-segment gloss
recognizer → gloss→text translator → agent). Each stage has its own failure mode and its own drift signal, so
monitoring and continual learning here are genuinely *multi-stage* and not a single-model accuracy curve.

Two facts shape everything below:

1. **The pose front-end is frozen, not trained.** MediaPipe Holistic (Apache-2.0, algorithmic) on Colab and the
   `SeedPoseEngine` offline are never updated by us. We monitor their *output quality* (landmark confidence,
   missing-frame rate) but we re-train only the two learnable units: the **per-segment gloss recognizer** and the
   **`google-t5/t5-small` gloss→text translator**.
2. **Drift in SLT is mostly distributional and demographic.** New signers, new cameras/lighting, new regional sign
   variants, and out-of-vocabulary signs degrade the system long before any weight rots. The monitoring strategy is
   therefore built around *coverage* and *abstention*, not just latency and error rate.

---

## 1. What we monitor

All monitoring is driven off the **structured job logs** every pipeline run already emits (one JSON record per
translate request, written by the standard logging/registry harness reused from P15/P17/P19/P20). The aggregator is
**`monitoring/drift_report.py`**, which rolls a window of job logs into a single drift report (JSON + the standard
chart set) and is wired into the automation/grading harness so it can run on a schedule or on demand.

Each job record carries, at minimum:

- `input_kind` — `seed` | `pose_frames` | `video` (which front-end path D1 routed to)
- `n_frames`, `n_segments` — frame count and how many sign units the D2 segmenter found
- `glosses`, `gloss_confidences` — per-segment recognized gloss + confidence
- `text` — assembled spoken-text output
- `low_conf_segments`, `low_conf_ratio` — segments below `recog_min_conf = 0.15` and their fraction
- `abstained` — whether D5 fired (`low_conf_ratio > oov_abstain_ratio = 0.5`)
- `verify_kept` — whether the D4 translation-verification round-trip / chrF keep-gate passed
- `latency_ms` per stage (`pose`, `segment`, `recognize`, `translate`) and end-to-end
- `pose_quality` — mean landmark-presence / detection confidence from the front-end (1.0 for `SeedPoseEngine`)

### 1.1 Core operational metrics

| Signal | Source field | Why it matters for P01 | Healthy band (synthetic baseline) |
|---|---|---|---|
| **Abstention rate** | `abstained` | The headline trust metric. A rising abstain rate means inputs are drifting away from the trained vocabulary/signer distribution. | Near 0 on clean synthetic; spikes on noisy/OOV input *by design* |
| **Low-confidence segment rate** | `low_conf_ratio` | Leading indicator of abstention — climbs *before* the system starts abstaining wholesale. | Low on clean input; the robustness sweep shows it climbing with pose noise |
| **Recognition-confidence drift** | mean & p10 of `gloss_confidences` | Falling mean gloss confidence = the recognizer is increasingly unsure → new signers/cameras or vocabulary gaps. | High and stable on in-distribution signs |
| **Mean segments per sentence** | `n_segments` | Sudden drops = the motion segmenter is under-splitting (signs blurring together, fast signing, low frame rate); spikes = over-splitting (jitter, dropped frames). | Tracks the 2–6 signs/sentence the synthetic generator produces |
| **Verification keep-rate** | `verify_kept` | A falling D4 keep-rate means gloss→text is producing outputs that don't round-trip — translator drift or OOV glosses. | High on lexicon-covered glosses |
| **Latency (per stage + e2e)** | `latency_ms` | Pose extraction dominates on real video; a creeping `pose` latency flags front-end / hardware regressions. | CPU offline path is fast; MediaPipe path is the real cost |

`drift_report.py` computes, per window: the rate of each boolean flag, the mean / p10 / p90 of each confidence and
latency series, the `n_segments` distribution, and a **per-gloss frequency table** (how often each of the 40 lexicon
glosses was recognized). The per-gloss table is the raw material for the coverage analysis in §2.

### 1.2 Pose-extraction quality (the frozen front-end)

Because MediaPipe Holistic is not trained by us, we cannot "improve" it — but we **must** watch it, since the entire
cascade is downstream of its keypoints. The keypoint layout (`pose/layout.py`, 2×21 hand + 25 body landmarks × 3
coords, mirroring `PSewmuthu/How2Sign_Holistic`) gives a per-frame presence signal. We track:

- **Missing-landmark rate** — frames where a hand or the upper body was not detected. High rates mean off-frame
  signing, occlusion, motion blur, or poor lighting. This is the single most common real-world degradation and it is
  *invisible* to a text-only monitoring mindset.
- **Mean detection confidence** — MediaPipe's own per-landmark confidence, aggregated per clip.
- **Frame-rate / resolution metadata** — low FPS directly breaks the motion segmenter (rest frames vanish; strokes
  alias), so we log it as a covariate.

On the offline `SeedPoseEngine` path these are trivially perfect (gold is embedded), so pose-quality monitoring is a
**Colab/production-only** signal — but it is logged with the same schema so the report code is identical in both
environments.

---

## 2. The vocabulary-coverage problem

This is the structural continual-learning challenge for P01, and it is sharper than for any of the text projects.

The offline system knows exactly **40 glosses** (`data/lexicon.py`, with its deliberately-non-identity gloss→text
map, e.g. `THANK-YOU`→"thank you", `ME`→"i"). The numpy nearest-centroid recognizer can only ever emit one of those
40 labels; the t5 translator was only ever trained on text for glosses it has seen. Any real deployment immediately
meets:

- **New signs (lexical OOV).** A sign outside the trained vocabulary cannot be recognized — at best the nearest
  centroid fires with low confidence, at worst it silently snaps to a wrong-but-confident neighbour. Sign languages
  are open-vocabulary (names, technical terms, regionalisms, classifiers), so OOV is the *normal* case, not an edge
  case.
- **New signers.** The same gloss has signer-specific articulation (handshape, speed, signing space). A recognizer
  fit to one signer set generalizes poorly — the acute **representation bias** flagged in the ethics doc.
- **New sign languages / dialects.** ASL ≠ ISL ≠ Icelandic SL. A model trained on one will fail wholesale on
  another; there is no graceful degradation.

### 2.1 Detecting coverage gaps from logs

The abstention machinery is *also* the coverage-gap detector. Concretely, `drift_report.py` surfaces:

- **OOV pressure** — the `low_conf_ratio` distribution and the abstain rate. A persistent population of segments just
  below `recog_min_conf` is the signature of recurring unknown signs.
- **Confidence-by-gloss** — glosses whose mean recognition confidence is drifting down are candidates for
  re-centroiding (new signer variation) or for being split into variants.
- **The "near-miss" log** — for abstained or low-confidence segments we retain the pose displacement and the runner-up
  centroid distances. Clusters of similar unrecognized displacements that *don't* match any centroid are very likely
  **a new sign the vocabulary should learn**.

### 2.2 Expanding the vocabulary

Closing a coverage gap is a deliberate, two-part operation — and crucially, **expanding the vocabulary requires
re-training the recognizer**, because adding a 41st gloss changes the label space:

1. **Add the gloss to the lexicon.** Extend `data/lexicon.py` with the new gloss and its gold spoken-text mapping.
   This is the source of truth for both stages.
2. **Re-fit the recognizer over the enlarged label set.** Offline, the numpy centroid classifier simply recomputes
   centroids including the new gloss's mean-displacement signature (cheap, deterministic, no torch). On Colab, the
   transformer recognizer is fine-tuned with the new class added to its head.
3. **Re-train / refresh the translator** so the new gloss token has a text mapping the t5 model can produce (see §4).
   Byte-level `google/byt5-small` is the fallback backbone here precisely because it is robust to never-before-seen
   gloss tokens.

The synthetic generator (`data/synth_pose.py`) makes this safe to rehearse: a new gloss gets a fresh deterministic
motion direction and immediately participates in generated sentences, so we can regression-test the enlarged
vocabulary end-to-end (segment → recognize → translate → agent) **before** touching any real-data path.

---

## 3. Feedback capture

The product surfaces (FastAPI `POST /translate`, the Gradio demo) are the feedback funnel. Because every output ships
with **per-sign confidence and an abstain flag**, corrections are naturally targeted at the segments the system was
already unsure about.

We capture three correction types, each of which becomes a different kind of training pair:

| Correction | Captured as | Feeds |
|---|---|---|
| **Wrong gloss** for a segment | `(pose_segment, corrected_gloss)` | the recognizer's training set (new centroid evidence / new class sample) |
| **Wrong text** for a correct gloss sequence | `(gloss_sequence, corrected_text)` | the t5 translator's parallel corpus |
| **Wrong / missed boundaries** | `(pose_window, corrected_boundaries)` | segmenter threshold tuning + boundary-F1 regression cases |

Capture discipline, given the **biometric / Deaf-community sensitivity** of this data:

- **Opt-in only, consented, and de-identified where possible.** Raw video is biometric and identifying. The default
  is to retain only the **pose-keypoint sequence** for the corrected segment (not the video) plus the gloss/text
  correction — keypoints are far less identifying than face/hand video, consistent with the on-device / no-retention
  posture in the privacy doc.
- **Abstained outputs are the gold mine.** When D5 abstains and a human interpreter then supplies the correct
  gloss/text, that pair is exactly an OOV or hard-signer example — the highest-value training data we can get.
- **Provenance + reviewer metadata** are stored with each pair (signer consent ref, sign language, source surface,
  whether a qualified Deaf signer / interpreter confirmed it) so re-training can weight or filter by trust and by
  language.
- **Human-in-the-loop is mandatory in medical/legal contexts.** Feedback there must come from a qualified
  interpreter, never auto-mined, because a wrong "confident" correction is worse than an abstention.

Captured pairs land in a versioned correction store and are converted into the same on-disk format the synthetic
generator emits (embedded-gold sequences for recognizer pairs; gloss→text rows for the translator corpus), so the
existing training/eval harness consumes real and synthetic data through one code path.

---

## 4. Periodic re-training

Re-training is **staged and independently triggered** — the two learnable units are decoupled, so we never pay for a
full retrain when only one stage drifted.

### 4.1 Recognizer (per-segment pose → gloss)

- **Offline (numpy nearest-centroid):** "re-training" is recomputing centroids from the union of synthetic sequences
  and the captured `(pose_segment, gloss)` corrections. It is deterministic, seconds-fast, CPU-only, and needs no
  torch — so it can run on essentially every coverage-expansion event.
- **Colab (transformer encoder over pose frames):** fine-tuned when corrections accumulate or a new gloss/signer set
  is added. The `manohonsy/how2sign-pose-cslr` reference (MIT, 4.8M params, pose+CTC CSLR) is our scale anchor — a
  ~5M-parameter pose model is student-scale and re-trainable on a single T4.
- **Trigger conditions:** abstain rate or `low_conf_ratio` crosses a threshold over a window; a vocabulary expansion
  (§2.2); a confirmed new-signer or new-camera cohort; a batch of accumulated wrong-gloss corrections.

### 4.2 Translator (`google-t5/t5-small`, gloss → text)

- Re-trained on the gloss→text parallel corpus (lexicon + captured text corrections) using the **P13/P14 seq2seq
  train/eval pattern** that is reused wholesale here.
- **Trigger conditions:** vocabulary expansion introducing gloss tokens the translator has never produced; a falling
  D4 verification keep-rate; accumulated wrong-text corrections; BLEU-4 / chrF regression on the held-out set.
- `facebook/m2m100_418M` (MIT, reused from P13/P14) is the multilingual alternative when target languages multiply;
  `google/byt5-small` (Apache-2.0) is preferred when the gloss vocabulary churns fast, since byte-level inputs absorb
  novel gloss tokens without a tokenizer rebuild.

### 4.3 Gate every retrain on the full metric suite — and stay honest

No re-trained model is promoted unless it holds or beats the incumbent on the standard evaluation, run on the
held-out synthetic split **and** the `Sigurdur/icelandic-sign-language` real-data smoke test (Apache-2.0, 214 rows —
the only cleanly permissive real corpus available):

- **Recognition:** gloss-WER (sub/del/ins) + position-aligned gloss accuracy + sequence exact-match.
- **Segmentation:** boundary-F1 vs gold sign boundaries.
- **Translation:** BLEU-1..4 (**BLEU-4 headline**) + chrF + WER.
- **Operational:** abstention rate and per-gloss confidence must not regress.
- **Baselines as guardrails:** the retrained core must still beat most-frequent gloss (~0.02 floor),
  random gloss, and identity-translate (gloss tokens as text, BLEU ~84 — the lexicon is worth ~15 BLEU), and
  approach the **Seed oracle** translate-stage upper bound.

> **Honesty caveat (carried into every retrain report).** Automatic SLT metrics — BLEU, chrF, ROUGE, BLEURT — are
> **unreliable**: length-sensitive and blind to hallucination and semantic equivalence (Yazdani et al. 2025,
> hf.co/papers/2510.25434). A BLEU bump alone never justifies a promotion. We pair the numbers with the abstention
> rate, per-sign confidence, and — for any safety-relevant deployment — qualified-signer / interpreter review of a
> sample. A model that scores higher but abstains less on genuinely OOV input has gotten *worse*, not better.

---

## 5. Drift taxonomy and responses

P01 drift is overwhelmingly **input-distribution drift**, and each axis maps to a specific log signal and response.

| Drift axis | Symptom in the logs | Primary response |
|---|---|---|
| **Signer distribution** (new people, articulation, speed) | falling mean gloss confidence; rising `low_conf_ratio` concentrated on specific glosses | re-fit/fine-tune recognizer with consented per-signer corrections; flag the representation-bias risk |
| **Camera / lighting / setup** | rising missing-landmark rate; falling MediaPipe detection confidence; pose latency drift | front-end is frozen — fix capture conditions (guidance, resolution/FPS), re-segment with adjusted thresholds; no model change |
| **Sign distribution** (new signs, regional variants, dialects) | OOV pressure; clusters of unmatched pose displacements in the near-miss log; rising abstain rate | vocabulary expansion (§2.2) → recognizer + translator retrain |
| **Pose-extraction quality** (blur, occlusion, low FPS) | under/over-segmentation (`n_segments` anomalies); missing-landmark spikes | tune the motion segmenter (rest-frame velocity threshold); surface a "please repeat / improve framing" note via the agent's advisory brain |
| **Translation drift** (new target phrasing, lexicon edits) | falling D4 verification keep-rate; chrF/BLEU regression | retrain the t5 translator on the refreshed parallel corpus |

Two safety properties make this loop trustworthy and are themselves monitored:

- **Abstention is the backstop.** Under any drift the system has not yet adapted to, D5 abstains rather than emitting
  a confident hallucination from noise — the core value-add over a blind end-to-end decode. A *drop* in abstention
  without a matching rise in input quality is itself a red flag worth investigating.
- **The advisory LLM brain stays off by default.** The optional `anthropic` brain only ever adds a "please repeat"
  note and never changes the output, so continual learning is driven by deterministic, auditable signals and
  human-confirmed corrections — never by an opaque online update.

### 5.1 The monitoring → learning loop, end to end

1. Every request logs the schema in §1 (offline and on Colab, identical fields).
2. `monitoring/drift_report.py` rolls the window into rates, drift series, the per-gloss table, and the near-miss
   clusters.
3. Threshold crossings (abstain rate, `low_conf_ratio`, confidence drift, verification keep-rate, segment-count
   anomalies, pose-quality drop) raise a review.
4. Consented, de-identified corrections — especially from abstained cases and interpreter review — are captured as
   recognizer / translator / segmenter training pairs.
5. Vocabulary is expanded (§2.2) and the affected stage(s) re-trained (§4), with the synthetic generator providing
   regression coverage for every new gloss.
6. The candidate is gated on the full recognition + segmentation + translation suite vs the baselines and the Seed
   oracle, **with the SLT-metric honesty caveat applied**, before promotion.

This keeps the **frozen pose front-end** stable, the **trainable core** current with the signing population it
actually serves, and the whole system honest about what it does not yet understand — abstaining and asking for a
human rather than guessing, exactly as the assistive (not replacement-for-interpreters) framing demands.
