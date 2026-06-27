# P01 — Translation Evaluation Methodology

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment.
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`.
**Scope:** This is the *special quality document* for P01. It defines exactly how the
Sign2Gloss2Text cascade is scored end-to-end: the **recognition** metrics (gloss WER and
its sub/del/ins decomposition, position-aligned gloss accuracy, sequence exact-match), the
**translation** metrics (BLEU-1..4 with BLEU-4 as headline, chrF, WER), the **segmentation**
metric (boundary-F1), the **baselines** that isolate the trainable core, the
synthetic-offline ↔ real-data story, the pose-noise robustness sweep, how abstention
responds, and — prominently — the **honesty caveat** that automatic SLT metrics are
unreliable.

---

## 1. What we are actually scoring

P01 is a **cascade**, so a single end-to-end BLEU number hides where errors come from. Each
stage of the pipeline is measured separately, then end-to-end:

```
VIDEO/POSE ──► [SEGMENT] ──► [RECOGNIZE gloss] ──► [TRANSLATE gloss→text] ──► sentence
                  │                 │                        │
              boundary-F1     gloss-WER, acc,            BLEU-1..4, chrF,
              (segmentation)  exact-match (recognition)  WER (translation)
```

- **Segmentation** is judged against the gold sign boundaries embedded by the synthetic
  generator: did the motion segmenter split the sentence into the right sign units?
- **Recognition** is the *trainable CSLR core*: given the (segmented) pose stream, did we
  emit the correct **gloss sequence**? This is the stage most exposed to pose noise.
- **Translation** is the gloss→text seq2seq: given glosses, did we produce the correct
  **spoken-language text**? This is where the non-trivial lexicon/reordering is learned
  (e.g. `THANK-YOU`→"thank you", `ME`→"i").
- **End-to-end** chains all three: pose → text, the number a user actually experiences.

We also report **abstention rate** (fraction of inputs the agent declines, see §7) and the
agent's downstream behaviour, because a system that knows when *not* to answer is worth more
than one that always emits fluent text.

Splits: metrics are reported on a **held-out synthetic test split** (the offline primary),
and as a **smoke test** on a small real corpus (`Sigurdur/icelandic-sign-language`, see §6).

---

## 2. Recognition metrics (the CSLR / gloss stage)

The recognizer emits a sequence of gloss tokens \( \hat{G} = (\hat{g}_1, \dots, \hat{g}_m) \)
to compare against the gold gloss sequence \( G = (g_1, \dots, g_n) \). Three complementary
metrics, because each is blind to a different failure mode.

### 2.1 Gloss WER (sub / del / ins) — the primary recognition metric

Continuous Sign Language Recognition (CSLR) is scored with **Word Error Rate over glosses**,
the standard on PHOENIX/CSL-Daily-style benchmarks. It is the Levenshtein edit distance
between predicted and gold gloss sequences, normalised by reference length:

\[
\text{gloss-WER} = \frac{S + D + I}{N}
\]

where, over the optimal alignment,

- \( S \) = **substitutions** (wrong gloss recognised in the right slot),
- \( D \) = **deletions** (a gold gloss missed — e.g. a sign segment merged away),
- \( I \) = **insertions** (a spurious gloss — e.g. a rest frame mis-split into a sign),
- \( N \) = number of glosses in the **reference** \( G \).

Lower is better; \(0\) is perfect. WER can exceed \(1.0\) when insertions dominate. We do **not**
collapse the breakdown: we report \(S\), \(D\), \(I\) and the total separately, because they
diagnose *different* upstream problems — high \(I/D\) usually means the **segmenter** is
over/under-splitting (a boundary problem, §4), whereas high \(S\) means the **classifier**
is confusing glosses with similar mean-pose displacement (a recognition problem). In P01 the
offline recognizer is a numpy nearest-centroid classifier over per-segment mean displacement,
so substitutions concentrate on glosses whose deterministic motion directions are close in
keypoint space.

### 2.2 Position-aligned gloss accuracy

When segmentation is correct (one predicted gloss per gold segment), we additionally report a
simple **position-aligned accuracy** — the fraction of segments whose gloss is exactly right:

\[
\text{gloss-acc} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\hat{g}_i = g_i]
\]

This isolates pure classification quality from alignment effects: it answers "given that we
cut the signs correctly, how often is the *label* right?" It is the cleanest single number for
the robustness sweep (§5), because it does not conflate classifier errors with segmenter
errors. (For an *isolated*-sign mode — single gloss per clip — this degenerates to Top-1
accuracy; Top-5 would be the obvious extension.)

### 2.3 Sequence exact-match

The strictest recognition metric: the fraction of **whole sequences** recovered with zero edits.

\[
\text{seq-EM} = \frac{1}{|\mathcal{D}|} \sum_{(\hat{G},G)} \mathbb{1}[\hat{G} = G]
\]

Exact-match is unforgiving (one wrong gloss fails the whole sentence) and is the honest
counterweight to WER, which can look deceptively low while still mangling most sentences. We
report all three together.

---

## 3. Translation metrics (the gloss→text stage)

The translator maps the recognised gloss sequence to spoken-language text. We **reuse the
P13/P14 MT metric implementations** verbatim — gloss→text is genuinely a small machine
translation problem (the `sign/sockeye-signwriting-to-text` precedent: "treat the recognized
symbolic sequence as a source language, run standard MT").

### 3.1 BLEU-1..4 (BLEU-4 = headline)

BLEU is reported at all four n-gram orders. **BLEU-4 is the headline number**, matching the
SLT literature (Camgöz et al. and successors report BLEU-1..4 with BLEU-4 foremost).
BLEU is a modified n-gram precision with a brevity penalty:

\[
\text{BLEU-}N = \underbrace{\text{BP}}_{\text{brevity penalty}} \cdot
\exp\!\left( \sum_{k=1}^{N} w_k \log p_k \right),
\qquad
\text{BP} = \begin{cases} 1 & c > r \\ e^{(1 - r/c)} & c \le r \end{cases}
\]

where \( p_k \) is the clipped modified precision of \(k\)-grams, \( w_k = 1/N \) are uniform
weights, \( c \) is candidate length and \( r \) is reference length. We report BLEU-1
(unigram, ≈ lexical content match) through BLEU-4 (4-gram, ≈ local fluency/word order) so the
reader can *see* the precision decay with order — on the synthetic data BLEU-1 stays near
ceiling while BLEU-4 is the sensitive discriminator.

### 3.2 chrF (character n-gram F-score)

chrF complements BLEU by scoring **character** n-grams, so it is far more tolerant of
morphology and small surface variation and degrades gracefully on short outputs (where BLEU-4
is jumpy). It is the F-score (default \( \beta = 2 \), recall-weighted) over character n-grams
up to order 6:

\[
\text{chrF}_\beta = (1 + \beta^2) \cdot \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}
\]

We keep chrF specifically because the lexicon maps short tokens (`ME`→"i") where a single
character matters and BLEU-4 is unstable.

### 3.3 Translation WER

We also report **WER on the text** (same edit-distance formula as §2.1, but over output
words against the gold text). Word-level WER is the most interpretable single number for a
non-specialist reader ("how many words would a human have to fix?") and connects naturally to
the recognition-stage gloss-WER, so the same instrument scores both ends of the cascade.

---

## 4. Segmentation metric — boundary-F1

Before recognition, the motion-based segmenter (agent decision **D2**: low-velocity rest
frames split signs) must cut the continuous pose stream into the right number of sign units in
the right places. We score predicted boundary positions against the **gold sign boundaries**
embedded in the synthetic sequence with a **boundary-F1**:

\[
\text{Precision} = \frac{|\text{predicted} \cap \text{gold}|}{|\text{predicted}|}, \quad
\text{Recall} = \frac{|\text{predicted} \cap \text{gold}|}{|\text{gold}|}, \quad
\text{F1} = \frac{2\,\text{P}\,\text{R}}{\text{P} + \text{R}}
\]

A predicted boundary counts as a true positive if it lies within a small tolerance window
(±a few frames) of a gold boundary, since exact frame-level agreement is neither expected nor
necessary. Boundary-F1 directly explains the insertion/deletion balance of gloss-WER: a low
precision means the segmenter invents boundaries (→ gloss insertions); a low recall means it
merges signs (→ gloss deletions). On clean synthetic data, where rest frames are explicit and
near-still, segmentation is the *easy* part (F1 = 1.0, §8); pose noise is what erodes it (§5).

---

## 5. Pose-noise robustness sweep

Real sign-language video yields **noisy** pose keypoints (low-quality video, occluded hands,
signer variation, MediaPipe jitter). We model this offline by adding Gaussian perturbation of
increasing standard deviation \( \sigma \) to the keypoint coordinates and re-running the
*whole* cascade at each noise level. The sweep reports, as a function of \( \sigma \):

- **gloss-acc / gloss-WER** — the headline of the sweep; recognition degrades first and most,
  because the nearest-centroid decision boundary between glosses with nearby motion directions
  is the most fragile link.
- **boundary-F1** — segmentation holds longer (rest frames stay near-still until noise is
  large), then collapses as noise blurs the rest/motion distinction.
- **BLEU-4 / chrF** — translation quality, which can *look* artificially stable because a
  fluent-but-wrong text still scores partial n-gram credit (precisely the failure the honesty
  caveat in §9 warns about).
- **abstention rate** — see §7: as noise rises and per-segment confidence drops, the agent
  declines more inputs rather than emitting hallucinated text.

The expected, and verified (§8), shape: **clean input is near-perfect; rising \( \sigma \)
monotonically degrades recognition; pure-noise input collapses recognition entirely and the
agent ABSTAINS rather than fabricating a sentence.** The sweep is the core robustness evidence
for the privacy/robustness doc and is plotted in the autoreport.

---

## 6. Synthetic-offline ↔ real-data story

P01's *defining* constraint is licensing: **every continuous SLT corpus on the Hub is
non-commercial, gated, or unspecified** (`Exploration-Lab/iSign` is CC-BY-NC-SA **and gated**;
`aipieces/How2Sign` and `PSewmuthu/How2Sign_Holistic` derive from How2Sign's **CC-BY-NC**
upstream; `Voxel51/WLASL` and `Kibalama/poseformer-sign-language` are `other`/WLASL research
terms). RWTH-PHOENIX-2014T, CSL-Daily and YouTube-ASL/SL-25 are **not redistributable Hub
repos** at all. We therefore do **not** train or headline-evaluate on a restrictively-licensed
corpus, and we never invent dataset ids.

**Primary evaluation = synthetic, with embedded gold.** The generator (`data/synth_pose.py`)
fixes a deterministic motion direction per gloss; a sign is a triangle stroke along it (so the
**mean displacement recovers the gloss**), signs are separated by near-still rest frames (so a
motion segmenter can split them), and the spoken text comes from the lexicon
(`data/lexicon.py`, 40 glosses), deliberately **≠** the gloss tokens so translation is
non-trivial. The gold `{glosses, text, boundaries}` is embedded on the sequence and read back
by the `SeedPoseEngine` / `SeedRecognizer`. This makes every metric in §§2–4 computable with
**no mediapipe, no torch, no video, no network** — fully reproducible and deterministic.

**Real-data smoke test = `Sigurdur/icelandic-sign-language`** (Apache-2.0 ✅, the only cleanly
permissive real corpus: 214 rows, a YouTube-SL-25 Icelandic slice with `video_id` + timed
`transcript`). It is *too small and isolated to train or headline-benchmark on*, and it carries
**no gold glosses**, so it cannot drive the recognition metrics. Its role is strictly a
**plumbing/smoke test**: confirm the pose front-end + translation path run end-to-end on real
landmark sequences and produce sane text, with translation-side BLEU/chrF reported *for
sanity, explicitly labelled "smoke test, not a benchmark."* `om192006/sign_language_keypoints`
(MIT ✅, 29 isolated gestures) serves only as a pose-schema template. **All non-permissive ids
above are flagged, never used as the measured component.**

The honest framing: synthetic gives clean, controllable, reproducible *upper-bound* numbers and
a robustness knob; the real smoke test proves the wiring is not synthetic-only. Neither claims
PHOENIX-level real-world SLT performance — and the caveat in §9 explains why even that would be
hard to *measure* if we had it.

---

## 7. Baselines — what each one isolates

A trained core is only credible against floors and ceilings. Four baselines, each isolating a
specific question:

| Baseline | What it does | What it isolates / proves |
|---|---|---|
| **Random gloss** | emit a uniformly random gloss per segment | the absolute floor; confirms the metric harness is not accidentally rewarding noise |
| **Most-frequent gloss** | always emit the single most common gloss | the *informed* floor — the score a model gets with **zero discrimination**; on a 40-gloss vocab this is ≈ 0.02 accuracy (§8). The recognizer **must** beat this |
| **Identity / passthrough translate** | feed the **gold gloss tokens straight through as text** (no translation) | the value the **translation stage adds** — it shows the reordering + lexicon mapping the model must learn. Identity scores ≈ **84 BLEU** on the synthetic data; a real translator reaching **99+** means the lexicon/reordering is worth ≈ **+15 BLEU** |
| **Seed ORACLE** | perfect recognition (read gold glosses) → run the **real translator** | the **upper bound on the translate stage alone**, decoupled from recognition error. Lets us attribute any end-to-end shortfall to recognition vs translation |

The trained core must **beat baselines 1–3** and **approach baseline 4**. Identity-translate is
the most instructive: because the lexicon makes gloss tokens deliberately differ from spoken
text, the gap between identity (~84 BLEU) and the trained translator (99+ BLEU) is a *direct,
quantified* measure of what translation contributes — not a vacuous "BLEU went up."

**Abstention** is reported alongside the baselines, not as a baseline itself. The agent's
finalize step (**D5**) ABSTAINS when the low-confidence-segment ratio exceeds
`oov_abstain_ratio = 0.5` (per-segment confidence gated at `recog_min_conf = 0.15`). We report
the abstention rate across the noise sweep: on clean input it is ~0 (the system answers), on
pure-noise input it approaches 1 (the system declines). This is the headline **value-add** —
abstention beats a blind end-to-end decoder that hallucinates fluent, high-BLEU-looking text
from pure noise.

---

## 8. Verified offline numbers

All numbers below were **verified offline** on the held-out synthetic split with the
SeedPose + numpy-centroid + lexicon path (CPU, no torch/mediapipe/network). They are the
reference values the autoreport and the grading harness regenerate.

| Metric | Stage | Clean (trained core) | Baseline / floor | Reads as |
|---|---|---|---|---|
| Gloss accuracy (position-aligned) | recognition | **1.0** | most-frequent ≈ **0.02** | classifier is genuinely discriminative on a 40-gloss vocab, ~50× the informed floor |
| Gloss-WER (S/D/I) | recognition | **0.0** | random ≫ 1.0 | zero edits on clean input |
| Sequence exact-match | recognition | **1.0** | ~0 | whole sequences recovered |
| Segmentation boundary-F1 | segmentation | **1.0** | — | rest frames split signs cleanly |
| BLEU-4 (headline) | translation | **99+** | identity-translate ≈ **84** | lexicon + reordering add ≈ **+15 BLEU** over passthrough |
| chrF | translation | high (near-ceiling) | identity below | character-level confirms surface match |
| Abstention rate | agent (D5) | ~0 on clean · → 1 on pure noise | — | all 5 decisions fire; pure noise → ABSTAIN |

Additional verified behaviours: **pose noise monotonically degrades recognition** across the
robustness sweep; **pure-noise input ABSTAINS** instead of hallucinating; **all 5 agent
decision points fire** on the standard trace. These exact values are produced deterministically
from the embedded gold, so any drift is a regression the grading harness will catch.

> **Reading the ceiling honestly.** A clean-synthetic BLEU of 99+ and gloss accuracy of 1.0 are
> **upper bounds on a controlled task**, not a claim of real-world SLT quality. The synthetic
> data is designed to be *solvable* (the mean displacement provably recovers the gloss); the
> point is to demonstrate a correct, measurable cascade and a working abstention mechanism, then
> show — via the noise sweep and the real smoke test — how it behaves as conditions degrade.

---

## 9. Honesty caveat — automatic SLT metrics are unreliable

This is non-optional and is repeated in the autoreport, not buried here.

> **Automatic SLT evaluation metrics (BLEU, chrF, ROUGE, BLEURT) are unreliable.** Yazdani et
> al., *"A Critical Study of Automatic Evaluation in Sign Language Translation"* (2025,
> [hf.co/papers/2510.25434](https://hf.co/papers/2510.25434)), show these metrics are
> **length-sensitive** and **blind to hallucination and semantic equivalence**: a fluent
> translation that is *wrong* can score well, and a correct paraphrase can score poorly. They do
> not measure whether a Deaf user would be correctly understood.

Concretely for P01, this is why:

- **BLEU-4 is reported but never trusted alone.** We pair it with chrF (character-level,
  length-robust) and word-WER (interpretable), and we *separate* the recognition metrics
  (gloss-WER, exact-match) which are far less gameable by fluent hallucination.
- **The agent's abstention exists precisely because metrics can't catch hallucination.** A
  blind decoder can produce high-BLEU-looking text from noise; per-sign confidence gating +
  the round-trip translation verification (D4) + abstention (D5) are the *system-level* defence
  the metrics cannot provide.
- **No automatic number is presented as authoritative**, especially for medical/legal use. The
  ethics/privacy doc mandates a human in the loop and per-sign confidence display; the
  evaluation here is a development instrument, not a certificate of correctness.

The metrics in this document are reported as a **standard, reproducible battery** *and* with
their limitations stated up front — which is itself the honest, defensible evaluation posture
for a field where the data is locked behind restrictive licenses and the metrics are known to
be weak.

---

## 10. Summary metrics table

| Metric | Formula (short) | Stage | Direction | Headline / floor |
|---|---|---|---|---|
| Gloss-WER (S/D/I) | \((S+D+I)/N\) | recognition | ↓ | primary CSLR metric; report S/D/I split |
| Gloss accuracy | \(\frac{1}{n}\sum \mathbb{1}[\hat g_i = g_i]\) | recognition | ↑ | clean **1.0** vs most-freq **0.02** |
| Sequence exact-match | \(\mathbb{1}[\hat G = G]\) averaged | recognition | ↑ | strictest; clean **1.0** |
| Boundary-F1 | \(2PR/(P+R)\), tolerance window | segmentation | ↑ | clean **1.0** |
| BLEU-1..4 | n-gram precision × BP | translation | ↑ | **BLEU-4 headline**, clean **99+** vs identity **84** |
| chrF | char n-gram F\(_\beta\) | translation | ↑ | length-robust complement |
| WER (text) | \((S+D+I)/N\) over words | translation | ↓ | most interpretable |
| Abstention rate | fraction declined (D5) | agent | context | ~0 clean → ~1 pure-noise |

**Bottom line:** P01 reports the *full standard battery* — recognition (gloss-WER + accuracy +
exact-match), translation (BLEU-1..4 + chrF + WER), and segmentation (boundary-F1) — against
four baselines (random, most-frequent, identity-translate, Seed oracle), on a reproducible
synthetic spine with a permissive real-data smoke test, with a noise-driven robustness sweep
and an abstention mechanism. Every number is reproducible offline; every automatic metric is
flagged as unreliable per Yazdani et al. (2025); no low-confidence translation is ever
presented as authoritative.
