# P01 — Sign Language Translation · Data Description

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment.
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/` · **Task:** Sign2Gloss2Text — translate a sign-language **video / pose-keypoint sequence** into spoken-language **text** via an intermediate **gloss** sequence.

This document describes the data that drives P01: **why a synthetic pose-sequence generator is the primary data source**, exactly **how that generator works**, the **keypoint layout** it shares with real MediaPipe output, the **real-world corpora** we evaluated (and their licenses, flagged loudly), and the **splits / sizes** used for training and evaluation.

> **License posture up front.** Every continuous sign-language-translation (SLT) corpus on the Hugging Face Hub that defines this task is **non-commercial, gated, or unspecified**. There is **no permissive, directly-loadable continuous-SLT corpus** available. This is not an oversight in our search — it is the **defining structural constraint of the SLT field**, and it is the reason P01's primary data is a reproducible synthetic generator rather than a downloaded benchmark.

---

## 1. Why synthetic pose data is the primary source

Unlike the 18 prior text / OCR projects in this assignment series, P01 cannot lean on a permissive public corpus. The data landscape for continuous SLT is uniformly restrictive, and we verified this directly on the Hub (authenticated as `ledinhminhquan`):

- **The task-defining corpora are non-commercial or gated.** `Exploration-Lab/iSign` — which literally defines the `SignPose2Text` task we are building — is **CC-BY-NC-SA *and* gated**. How2Sign (the most-cited ASL→English continuous corpus) is **CC-BY-NC** upstream, and every Hub mirror inherits that.
- **The "permissive" real corpora are tiny and/or isolated-sign, not continuous SLT.** The only cleanly Apache-licensed real corpus we found, `Sigurdur/icelandic-sign-language`, has **214 rows**; the only clean MIT keypoint set, `om192006/sign_language_keypoints`, covers **29 isolated gestures**. Neither is large enough or continuous enough to train and measure a Sign2Gloss2Text cascade.
- **The gold-standard academic benchmarks are not redistributable.** RWTH-PHOENIX-Weather-2014T, CSL-Daily, and YouTube-ASL / YouTube-SL-25 live on university servers under academic-use licenses; they are **not clean, redistributable Hub repos** and we deliberately **do not invent ids** for them.

A controlled synthetic generator is therefore not a fallback — it is the **defensible, honest, reproducible** primary spine, and it is the right engineering choice for several concrete reasons:

1. **It is fully permissive and shippable.** The generator is our own code (no license encumbrance), so the entire offline pipeline — generate → segment → recognize → translate → evaluate → agent → tests — runs with **no MediaPipe, no torch, no video, and no network**, and can be redistributed, graded, and reproduced bit-for-bit.
2. **It embeds gold labels we control.** Because we synthesize the motion, we know the exact `{glosses, text, boundaries}` for every sequence. That gives us perfectly clean supervision and lets a `SeedPoseEngine` / `SeedRecognizer` read the gold back for oracle baselines — impossible to do reliably with noisy, partially-labelled real video.
3. **It isolates the trainable core.** The cascade's *only* trained component is the sign→gloss recognizer + gloss→text translator. A generator with known motion directions and a known gloss→text lexicon lets us prove the recognizer genuinely classifies sign motion (not noise) and that the translator genuinely learns reordering/lexicalization (not identity copying).
4. **It mirrors the established offline pattern** used in P15 / P17 / P19 / P20 of this series: an embedded-gold synthetic generator + a Seed/Stub offline engine, upgraded to real models on Colab.
5. **It produces the *same data shape* as real MediaPipe Holistic output** (see §3), so the Colab path — real MediaPipe Holistic on real video, a transformer recognizer, a `t5-small` translator — is a drop-in upgrade, not a rewrite.

The synthetic spine handles **training, the held-out evaluation split, baselines, robustness sweeps, and the agent's abstention logic**. The permissive real corpora (§4) are used as a **real-data smoke test** to confirm the schema and front-end work on genuine signing, not as the measured training set.

---

## 2. How the synthetic pose-sequence generator works

The generator lives in `data/synth_pose.py` and draws its vocabulary from `data/lexicon.py`. It produces **pose-keypoint trajectories** (sequences of per-frame keypoint vectors), *not* images or rendered video — the front-end's job (MediaPipe) is assumed already done, so the synthetic data starts at the pose level.

### 2.1 The 40-gloss lexicon and the gloss→text map

`data/lexicon.py` defines a small vocabulary of **40 ASL-style glosses** (e.g. `THANK-YOU`, `ME`, `YOU`, `HELLO`, `NAME`, `GOOD`, ...). Critically, each gloss maps to spoken text that is **deliberately different from the gloss token itself**, so that gloss→text translation is a non-trivial learned step rather than an identity copy:

| Gloss (source) | Spoken text (target) |
|---|---|
| `THANK-YOU` | `thank you` |
| `ME` | `i` |
| `YOU` | `you` |
| `GOOD` | `good` |

This lexicon gap is intentional and measurable: an **identity baseline** that emits the gloss tokens verbatim scores meaningfully below a model that has learned the lexicon (verified offline: identity-translate ≈ BLEU 84, the trained lexicon translator ≈ BLEU 99+, so the lexicon contributes ≈ 15 BLEU). It also injects ASL-vs-English reordering/morphology the translator must learn (`ME` → `i`, dropped articles, etc.).

### 2.2 One sign = a directed triangle stroke

For each gloss the generator fixes a **deterministic motion direction** in keypoint space, **seeded by the gloss index** (so the mapping is reproducible and stable across runs). A single sign is rendered as a **triangle stroke** — a smooth out-and-back excursion of the active keypoints along that gloss's direction. The key property is that the **per-sign mean keypoint displacement points along the gloss's signature direction**, so:

- the offline recognizer (a pure-numpy **nearest-centroid classifier**) can recover the gloss from the segment's mean-displacement vector — it is genuinely classifying the sign's motion, with **no torch involved**;
- distinct glosses occupy distinct directions in keypoint space, making the classes separable on clean data and **gracefully degradable under noise** (the robustness sweep below).

### 2.3 Rest frames separate signs (so segmentation is real)

Consecutive signs are separated by short runs of **near-still "rest" frames** (very low inter-frame velocity). This makes the **motion-based segmenter** a genuine component: it detects low-velocity rest regions and splits the stream into sign units there. Boundaries are not given to the segmenter at inference — it must find them — which is what makes **boundary-F1 vs gold boundaries** a meaningful segmentation metric.

### 2.4 A sentence = 2–6 concatenated signs

A synthetic **sentence** concatenates **2 to 6 signs** (drawn from the lexicon), interleaved with rest frames, producing a continuous pose stream that looks like uninterrupted signing. The spoken-text target is the per-gloss lexicon text assembled across the sentence (e.g. `ME THANK-YOU YOU` → `i thank you you`). This gives the cascade a realistic continuous-SLR-then-MT problem at small scale.

### 2.5 Embedded gold + the SeedPoseEngine

Each generated sequence carries an **embedded gold spec** `{glosses, text, boundaries}` attached to the sequence itself. Two offline engines read it back:

- **`SeedPoseEngine`** — the offline stand-in for MediaPipe Holistic. Offline it does not run any vision model; it returns the synthetic pose sequence (and can surface the embedded gold) so the rest of the pipeline runs with zero heavy dependencies. On Colab this role is filled by **real MediaPipe Holistic** on real video frames.
- **`SeedRecognizer`** — reads the embedded gold glosses to provide a **perfect-recognition oracle** (the upper bound on the translate stage), against which the real numpy nearest-centroid recognizer is compared.

This embedded-gold design is what lets the **whole pipeline — segment → recognize → translate → eval → agent → tests — run with NO mediapipe / torch / video / network**.

### 2.6 Behaviour verified offline

The generator and the offline core have been exercised end-to-end. On clean synthetic data:

- gloss recognition **accuracy = 1.0**, gloss-sequence translation **BLEU 99+**, segmentation **boundary-F1 = 1.0**;
- the **most-frequent-gloss floor ≈ 0.02** accuracy and the **identity-translate baseline ≈ BLEU 84** (so the lexicon adds ≈ 15 BLEU), confirming the task is non-trivial and the trained core earns its score;
- a **pose-noise robustness sweep** degrades recognition smoothly as injected keypoint noise rises (the centroid classifier is sensitive to motion corruption, as intended);
- **pure-noise input triggers abstention** — the agent declines rather than hallucinating fluent text;
- **all 5 agent decision points fire** across the generated test set.

---

## 3. Keypoint layout (mirrors MediaPipe Holistic / How2Sign_Holistic)

The per-frame keypoint vector is defined in `pose/layout.py` and is **shape-compatible with real MediaPipe Holistic output**, so synthetic-trained code transfers to real pose data without reshaping:

| Group | Landmarks | Coords each | Subtotal |
|---|---|---|---|
| Left hand | 21 | 3 (x, y, z) | 63 |
| Right hand | 21 | 3 (x, y, z) | 63 |
| Body (pose) | 25 | 3 (x, y, z) | 75 |
| **Per-frame vector** | **67** | **3** | **201 values** |

- **2 × 21 hand landmarks** match MediaPipe's per-hand 21-point hand mesh.
- **25 body landmarks** correspond to the upper-body subset of MediaPipe Holistic's pose landmarks most relevant to signing (face landmarks are excluded from the trajectory model by default; their use is constrained by the biometric/privacy posture in the ethics doc).
- **× 3 coordinates** (x, y, z) per landmark, normalized in MediaPipe's frame-relative coordinate space.

This is the **same shape produced by MediaPipe Holistic and by `PSewmuthu/How2Sign_Holistic`** (which stores `.npy` Holistic landmark sequences for How2Sign), and the same family of pre-extracted keypoints as `om192006/sign_language_keypoints`. A pose *sequence* is then a `T × 201` array (T frames), which is exactly what the segmenter, the recognizer, and the Colab transformer consume.

---

## 4. Real-world corpora evaluated (licenses flagged)

These corpora were assessed during stack verification. **None is used as the measured training set**; the permissive ones serve as a real-data smoke test, and the restrictive ones are documented to justify the synthetic spine. Restrictive licenses are flagged loudly below.

| Hub id | What it is | License | Status for P01 |
|---|---|---|---|
| `Sigurdur/icelandic-sign-language` | YouTube-SL-25 Icelandic slice; 214 rows, `video_id` + timed `transcript` | **Apache-2.0** ✅ permissive | **Real-data smoke test** — the ONLY cleanly permissive real corpus; too small/continuous-thin to train on |
| `om192006/sign_language_keypoints` | Pre-extracted MediaPipe keypoints, 29 isolated gestures | **MIT** ✅ permissive | Pose-schema template / sanity check; isolated, not continuous |
| `PSewmuthu/How2Sign_Holistic` | MediaPipe Holistic landmark sequences (`.npy`) for How2Sign | repo tagged MIT | ⚠️ **FLAG — derived from How2Sign, whose upstream is CC-BY-NC (non-commercial). Treat as non-commercial regardless of the repo tag.** |
| `aipieces/How2Sign` | How2Sign ASL→English video + keypoints | **unspecified** | ⚠️ **FLAG — unspecified = all-rights-reserved by default; upstream How2Sign is CC-BY-NC (non-commercial).** |
| `Exploration-Lab/iSign` | Indian Sign Language; defines the `SignPose2Text` task (this exact task), 118K+ pairs | **CC-BY-NC-SA + GATED** | 🚩 **FLAG — non-commercial + ShareAlike + GATED (access request required). Cannot be used for a commercial / freely-redistributable deliverable.** |
| `Kibalama/poseformer-sign-language` | WLASL → MediaPipe landmarks | **unset** (WLASL upstream is "other") | ⚠️ **FLAG — isolated signs; non-permissive WLASL source; license unset.** |
| `Voxel51/WLASL` | 11,980 isolated-sign videos | **other** (research terms) | ⚠️ **FLAG — research-only terms; isolated recognition only, not continuous SLT.** |

**Not on the Hub — do NOT invent ids.** RWTH-PHOENIX-Weather-2014T, CSL-Daily, and YouTube-ASL / YouTube-SL-25 (as full redistributable repos) live on university servers under academic-use licenses. `Exploration-Lab/ISLTranslate` does **not** exist on the Hub — the correct id is `Exploration-Lab/iSign`.

### License legend

- ✅ **Permissive** (Apache-2.0 / MIT) — usable, but only available here in tiny/isolated form.
- 🚩 **Non-commercial + gated** (CC-BY-NC-SA, access-gated) — strongest restriction; flagged.
- ⚠️ **Non-commercial / unspecified / other** — non-commercial upstream, all-rights-reserved-by-default, or research-only terms; flagged.

---

## 5. Splits and sizes

### 5.1 Primary (synthetic) data

The synthetic generator produces data on demand from a fixed random seed, so sizes are **configurable and reproducible** rather than fixed downloads. The default offline configuration is:

| Split | Composition | Purpose |
|---|---|---|
| **Train** | Synthetic sentences (2–6 signs each, drawn from the 40-gloss lexicon), generated from the training seed | Fit the gloss→text translator (and, on Colab, the transformer recognizer); fit the numpy centroid classifier offline |
| **Dev** | Held-out synthetic sentences (disjoint seed) | Tune thresholds (`recog_min_conf`, `oov_abstain_ratio`, segmentation velocity gate) |
| **Test** | Held-out synthetic sentences (disjoint seed) | Report all headline metrics: gloss-WER, position-aligned gloss accuracy, sequence exact-match, BLEU-1..4 (BLEU-4 headline), chrF, translation-WER, segmentation boundary-F1, abstention rate |

- **Vocabulary:** 40 glosses (`data/lexicon.py`), each with a distinct spoken-text mapping.
- **Sentence length:** 2–6 signs, separated by rest frames.
- **Per-frame dimensionality:** 201 values (67 landmarks × 3 coords); a sequence is `T × 201`.
- **Reproducibility:** splits are seed-disjoint; regenerating with the same seeds reproduces the exact data, labels, and metrics.

### 5.2 Robustness / stress data

Derived from the test split by **injecting keypoint noise** at increasing magnitudes (the robustness sweep), plus a **pure-noise** condition that should trigger agent abstention. These probe noisy-pose robustness and the abstention path, not headline accuracy.

### 5.3 Real-data smoke-test data

| Source | Size | Use |
|---|---|---|
| `Sigurdur/icelandic-sign-language` | 214 rows (`video_id`, timed `transcript`) | Confirm the MediaPipe front-end + keypoint schema + segmenter run on genuine signing; **not** used to compute headline metrics |
| `om192006/sign_language_keypoints` | 29 isolated gestures (pre-extracted keypoints) | Validate the `T × 201`-style keypoint schema against an external pose set |

These are smoke tests only: too small (214 rows) and too isolated (29 gestures) to train or fairly benchmark a continuous Sign2Gloss2Text cascade, and licensing prevents any larger continuous corpus from filling the gap.

---

## 6. Summary

- **Primary data = a synthetic pose-sequence generator** (`data/synth_pose.py` + `data/lexicon.py`): 40 glosses, deterministic per-gloss motion directions, triangle strokes, rest-frame separators, 2–6-sign sentences, a deliberately non-identity gloss→text lexicon, and **embedded gold** `{glosses, text, boundaries}` read back by `SeedPoseEngine` / `SeedRecognizer`. It runs the full pipeline with **no MediaPipe / torch / video / network**.
- **Keypoint layout** = 2 × 21 hand + 25 body landmarks × 3 coords = **201 values/frame**, **shape-identical to MediaPipe Holistic / `PSewmuthu/How2Sign_Holistic`**, so the Colab real-video path is a drop-in upgrade.
- **Real corpora** were evaluated but are **non-commercial, gated, or unspecified** (iSign CC-BY-NC-SA + gated 🚩; How2Sign / its Hub mirrors CC-BY-NC ⚠️; WLASL "other" ⚠️), with only **tiny permissive exceptions** (`Sigurdur/icelandic-sign-language` Apache, 214 rows ✅; `om192006/sign_language_keypoints` MIT, 29 gestures ✅) used as real-data smoke tests.
- The **synthetic spine is the honest, reproducible, license-clean answer** to the defining constraint of the SLT field — restrictive data — and is explicitly designed to transfer to real MediaPipe pose data without changing the data shape.
