# P01 — Sign Language Translation · Agent Architecture

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment.
**Package:** `signlang` · **Module:** `src/signlang/agent/`.

This document specifies the **deterministic finite-state machine (FSM)** that drives the Sign Language
Translation pipeline — the mandatory *agentic* component of P01. The agent does not "think" with a language
model in its critical path; it is a transparent, replayable controller that wires together the
Sign2Gloss2Text cascade (ingest → segment → recognize → translate → finalize) with **five explicit decision
points**, a structured **ToolTrace** audit, and a **fail-soft abstention** policy. Its value is not raw
accuracy but *calibrated honesty*: it knows when not to answer.

---

## 1. Why an agent (and not a single decode)?

A naive SLT system is one tensor op: feed the whole pose sequence into a seq2seq model and read out fluent
text. That design has a fatal property for this domain — **it always produces a confident, grammatical
sentence, even from noise, even from signs the model has never seen.** Sign-language data is biometric,
Deaf-community data; representation bias is acute (a model trained on one sign language / signer set fails on
others); and the downstream consumer may be a clinician or a court. A pipeline that hallucinates a plausible
translation from an out-of-vocabulary sign is not just wrong — it is dangerous.

The agent exists to interpose **structure and refusal** between the pose stream and the answer:

1. **Segmentation** — split the continuous stream into discrete sign units *before* recognition, so each
   sign is judged on its own evidence rather than smeared into a single global guess.
2. **Per-sign confidence gating** — attach a confidence to every recognized gloss and flag the weak ones.
3. **Translation verification** — round-trip the produced text back to gloss and keep it only if it agrees
   with what was recognized (a chrF keep-gate), catching translator hallucination.
4. **Abstention** — if too much of the sentence is low-confidence or out-of-vocabulary, **decline to answer**
   ("uncertain" + `needs_review`) instead of emitting fluent garbage.

This is the explicit **value-add**: *segmentation + confidence gating + verification + abstention* — a system
that beats a blind end-to-end decode precisely on the inputs where a blind decode is most harmful.

---

## 2. The FSM at a glance

The agent is a linear five-state machine with one terminal branch at each state. States run in order; any
state may **fail-soft** to the terminal `ABSTAIN`/`FAIL` sink rather than crash. Every transition appends a
`ToolTrace` record.

```
            D1                  D2                   D3                      D4                       D5
 INGEST ─────────► SEGMENT ─────────► RECOGNIZE ─────────► TRANSLATE+VERIFY ─────────► FINALIZE
   │ frame-count      │ motion           │ per-sign            │ gloss→text +            │ OOV / low-conf
   │ gate +           │ velocity          │ gloss +            │ round-trip               │ ratio gate
   │ route            │ rest-frame         │ confidence         │ chrF keep-gate          │
   ▼                  ▼ split             ▼                    ▼                         ▼
 [fail: too few    [fallback: single   [flag low-conf       [re-flag segments that    ANSWER ◄─┬─► ABSTAIN
  frames → FAIL]    span if no rests]    segments]            fail verification]       glosses+   "uncertain"
                                                                                       text+conf  +needs_review
```

- **State container:** a single mutable `AgentState` dataclass threads through all five states
  (`frames`, `segments`, `glosses`, `confidences`, `text`, `flags`, `trace`, `decision`).
- **Determinism:** given the same input and the same `AgentConfig`, the agent produces a byte-identical
  trace and output. No randomness, no wall-clock, no network in the offline path.
- **Offline first:** with the `SeedPoseEngine` + numpy nearest-centroid `SeedRecognizer` + lexicon
  translator, the entire FSM runs with **no mediapipe, no torch, no video, no network**.

---

## 3. AgentConfig — the thresholds

All decision thresholds live in one place (`src/signlang/agent/config.py`, an `AgentConfig` dataclass) so
the policy is auditable and reproducible. Defaults:

| Field | Default | Used at | Meaning |
|---|---|---|---|
| `min_frames` | `8` | D1 | minimum frame count for a usable pose sequence; below this the input is rejected |
| `motion_quantile` | `0.30` | D2 | per-frame velocity at/below this quantile is treated as a "rest" frame; rest runs split signs |
| `recog_min_conf` | `0.15` | D3 | per-segment gloss confidence below this is flagged **low-confidence** |
| `verify_chrf_min` | `0.20` | D4 | round-trip chrF (recognized gloss vs back-translated gloss) below this **re-flags** the segment |
| `oov_abstain_ratio` | `0.50` | D5 | if (low-conf + OOV + failed-verify) segments / total > this, the agent **ABSTAINS** |
| `min_rest_run` | `2` | D2 | a rest gap must span at least this many consecutive frames to count as a boundary |
| `use_llm_brain` | `False` | (advisory) | optional `anthropic` brain; OFF by default, never alters output |

> Thresholds are tuned on the held-out synthetic split. They are *policy*, not learned weights — changing
> `oov_abstain_ratio` changes how cautious the agent is, not what it recognizes. Every threshold that fires
> is written to the trace with the observed value beside it, so a reviewer can see *why* the agent abstained.

---

## 4. The five decision points

Each decision below lists: the **intermediate signal** it inspects, the **branches** it can take, the
**threshold** from `AgentConfig`, and the **fail-soft** behavior.

### D1 — Ingest: frame-count gate + route

- **State:** `INGEST`
- **Signal:** the length of the pose-keypoint sequence, `n_frames = len(frames)`, plus the input *kind*
  (`seed` reference, raw `frames`/pose array, or `video`).
- **Routing:** if the input is a video, it is sent through the front-end
  (MediaPipe Holistic on Colab → pose sequence; `SeedPoseEngine` offline reads the gold embedded in the
  synthetic sequence). If the input is already a pose array, it passes through unchanged. The pose layout is
  fixed by `pose/layout.py`: 2×21 hand + 25 body landmarks × 3 coords per frame — the same shape MediaPipe
  Holistic and `PSewmuthu/How2Sign_Holistic` produce.
- **Gate:** `n_frames >= min_frames` (default 8).
- **Branches:**
  - **proceed** → frames are normalized and handed to D2.
  - **fail** → `n_frames < min_frames`: the agent fail-softs to `FAIL` with reason
    `"too_few_frames"`, returns `decision="uncertain"`, `needs_review=True`, no text. (An empty or 1–2 frame
    clip cannot contain a sign; refusing here is correct.)
- **Trace:** `{"state":"ingest","n_frames":N,"kind":"seed|frames|video","min_frames":8,"pass":true|false}`.

### D2 — Segment: motion-based sign segmentation

- **State:** `SEGMENT`
- **Signal:** the **per-frame velocity** `v[t] = ||pose[t] − pose[t−1]||` over the keypoint vector. In the
  synthetic generator each sign is a triangle stroke along a fixed motion direction, separated by near-still
  **rest** frames — so velocity is high *inside* a sign and ~0 *between* signs.
- **Threshold:** frames with `v[t] <= quantile(v, motion_quantile)` (default 0.30) are **rest frames**. A run
  of `>= min_rest_run` (default 2) consecutive rest frames is a **boundary**; the spans between boundaries are
  the segments.
- **Branches:**
  - **n segments** → one or more sign units detected; passed to D3 as a list of frame spans.
  - **single span** (fallback) → no rest run found (continuous motion, or a single sign): the whole sequence
    is treated as one segment. This is a *soft* fallback, not a failure — it simply means the segmenter could
    not split the stream.
- **Fail-soft:** never crashes; worst case is one over-long segment, which D3 will likely recognize with low
  confidence, propagating caution downstream.
- **Trace:** `{"state":"segment","velocity_q30":q,"rest_frames":k,"n_segments":m,"boundaries":[...]}`.

### D3 — Recognize: per-segment gloss + confidence gate

- **State:** `RECOGNIZE`
- **Signal:** for each segment, the **mean-pose displacement** (offline) or the transformer-encoder logits
  (Colab). Offline, the `SeedRecognizer` is a **pure-numpy nearest-centroid classifier**: it computes the
  segment's mean displacement vector and assigns the nearest gloss centroid; *confidence* is a softmax-style
  margin between the top-1 and runner-up centroid distances. (Reference scale: `manohonsy/how2sign-pose-cslr`,
  MIT, 4.8M params, proves a ~5M pose model is student-scale.) On Colab this upgrades to a compact transformer
  encoder over pose frames + a `t5-small` translator.
- **Threshold:** `recog_min_conf` (default 0.15).
- **Branches (per segment):**
  - **confident** → `conf >= recog_min_conf`: gloss kept as-is.
  - **low-conf** → `conf < recog_min_conf`: gloss is **flagged** (not dropped — it still contributes a
    placeholder so positions stay aligned for WER). A gloss not in the 40-entry lexicon (`data/lexicon.py`)
    is additionally tagged **OOV**.
- **Output:** a parallel list of `(gloss, confidence, flags)` — one per segment — preserving order.
- **Trace:** per segment `{"state":"recognize","seg":i,"gloss":"THANK-YOU","conf":0.91,"low_conf":false,"oov":false}`.

### D4 — Translate + verify: gloss→text + round-trip keep-gate

- **State:** `TRANSLATE+VERIFY`
- **Signal (translate):** the recognized gloss sequence → spoken text. Offline this is the **lexicon
  translator** (gloss→text map, deliberately ≠ gloss tokens: `THANK-YOU`→"thank you", `ME`→"i"); on Colab it
  is `google-t5/t5-small` (Apache-2.0, the default) — with `facebook/m2m100_418M` (MIT, reused from P13/P14)
  and `google/byt5-small` (Apache-2.0, byte-level, robust to OOV/symbolic glosses) as alternates. This mirrors
  the `sign/sockeye-signwriting-to-text` (MIT) precedent: *treat the recognized symbolic sequence as a source
  language and run standard MT.*
- **Signal (verify):** a **round-trip agreement check**. The produced text is back-mapped to gloss
  (text→gloss) and compared to the recognized gloss via **chrF**. Low agreement means the translator
  invented content not supported by the recognized signs.
- **Threshold:** `verify_chrf_min` (default 0.20).
- **Branches:**
  - **keep** → round-trip chrF `>= verify_chrf_min`: text accepted.
  - **re-flag** → chrF `< verify_chrf_min`: the offending segment(s) are marked `verify_failed` and counted
    toward the abstain ratio in D5. The text is *retained* but no longer trusted on its own.
- **Why it matters:** automatic SLT metrics (BLEU/chrF/ROUGE/BLEURT) are themselves unreliable —
  length-sensitive and blind to hallucination/semantic equivalence (Yazdani et al. 2025,
  hf.co/papers/2510.25434). The round-trip is a *cheap internal consistency* check, not a quality guarantee;
  it is used only to *gate trust*, never reported as an accuracy number.
- **Trace:** `{"state":"translate","text":"thank you","roundtrip_chrf":0.74,"verify_min":0.20,"verify_failed":false}`.

### D5 — Finalize: abstain on OOV / low-confidence

- **State:** `FINALIZE`
- **Signal:** the **bad-segment ratio** = (low-conf ∪ OOV ∪ verify_failed segments) / total segments.
- **Threshold:** `oov_abstain_ratio` (default 0.50).
- **Branches:**
  - **answer** → ratio `<= oov_abstain_ratio`: return `decision="answer"`, the gloss sequence, the spoken
    text, and **per-sign confidence**. Individual flagged signs are still surfaced (per-sign confidence is
    always shown) so the consumer sees *which* signs were weak.
  - **abstain** → ratio `> oov_abstain_ratio`: return `decision="uncertain"`, `needs_review=True`, and a
    human-readable note. The text is withheld from the headline result (or clearly marked low-trust). This is
    the load-bearing branch: on pure-noise input the recognizer is uniformly low-confidence, the ratio → 1.0,
    and the agent **abstains** instead of hallucinating.
- **Trace:** `{"state":"finalize","bad_ratio":0.67,"abstain_ratio":0.50,"decision":"uncertain","needs_review":true}`.

---

## 5. Decision table

| # | State | Intermediate signal | Threshold (AgentConfig) | Pass branch | Fail / flag branch |
|---|---|---|---|---|---|
| **D1** | ingest | frame count `n_frames`; input kind | `min_frames = 8` | proceed → route video/pose to D2 | `FAIL "too_few_frames"` → uncertain + needs_review |
| **D2** | segment | per-frame velocity `‖Δpose‖`; rest runs | `motion_quantile = 0.30`, `min_rest_run = 2` | *n* segments → D3 | single-span fallback (no rest run) |
| **D3** | recognize | mean-pose displacement → gloss + margin confidence | `recog_min_conf = 0.15` | confident gloss kept | flag low-conf / OOV (kept, position-aligned) |
| **D4** | translate+verify | gloss→text; round-trip text→gloss chrF | `verify_chrf_min = 0.20` | keep text | re-flag segment `verify_failed` |
| **D5** | finalize | bad-segment ratio (low-conf ∪ OOV ∪ verify_failed) | `oov_abstain_ratio = 0.50` | **answer** + glosses + text + per-sign conf | **ABSTAIN** "uncertain" + needs_review |

---

## 6. ToolTrace audit

Every state appends one immutable record to `state.trace`, a list of `ToolTrace` dataclasses
(`src/signlang/agent/trace.py`):

```
ToolTrace(
    step:      int,          # 0..4, the decision index
    state:     str,          # "ingest" | "segment" | "recognize" | "translate" | "finalize"
    signal:    dict,         # the observed intermediate value(s): n_frames, velocity quantile, conf, chrF, ratio
    threshold: dict,         # the AgentConfig value(s) the signal was compared against
    branch:    str,          # the branch taken, e.g. "proceed" | "low_conf" | "abstain"
    note:      str,          # human-readable one-liner
)
```

Properties:

- **Replayable:** the trace is sufficient to reconstruct *why* the agent reached its decision — every gate
  records both the observed signal and the threshold it was compared against.
- **Serializable:** the full trace is returned by the API (`POST /translate`) and embedded in the autoreport,
  so a reviewer never has to re-run the agent to understand a result.
- **Privacy-aware:** the trace stores *derived signals* (counts, velocities, confidences, glosses), never raw
  landmark coordinates or video — consistent with the no-retention-by-default biometric policy.

---

## 7. Optional LLM brain (off by default, advisory only)

The agent supports an optional `anthropic` LLM "brain" (`use_llm_brain=False` by default). When enabled it is
**purely advisory**: it receives the trace summary (n segments, flagged signs, abstain ratio) and may emit a
natural-language note — e.g. *"Several signs were unclear; please repeat the sentence more slowly."* — to be
shown alongside an abstention.

Hard guarantees:

- The brain **never changes the glosses, the text, or the decision.** The FSM output is computed entirely by
  the deterministic states; the brain only annotates.
- It is **off by default** for privacy (no biometric-derived signals leave the device unless the operator
  explicitly opts in) and for determinism (graded/offline runs must be reproducible with no network).
- If the brain errors or is unavailable, the agent proceeds unchanged — it is a strict add-on, never a
  dependency.

---

## 8. Fail-soft behavior

The agent treats *every* abnormal condition as a reason to **abstain and flag**, never to crash or to guess:

| Condition | Where caught | Result |
|---|---|---|
| Empty / too-short clip | D1 | `FAIL "too_few_frames"` → uncertain + needs_review |
| Continuous motion, no rest gaps | D2 | single-span fallback (degrades gracefully, propagates low conf) |
| Unseen / OOV sign | D3 | flagged OOV, position-aligned placeholder, counts toward abstain |
| Translator hallucination | D4 | round-trip chrF fails → `verify_failed`, counts toward abstain |
| Mostly-noise input | D5 | bad ratio > `oov_abstain_ratio` → **ABSTAIN** |
| LLM brain error | (advisory) | ignored; deterministic output unchanged |

Fail-soft is the operational expression of the ethics stance: an **assistive aid, not a replacement for a
human interpreter**. A low-confidence translation is never presented as authoritative; the agent abstains,
shows per-sign confidence, and routes to a human in the loop — especially critical in medical/legal settings.

---

## 9. Worked example trace

**Input:** a synthetic pose sequence for the sentence `THANK-YOU ME` (gold text: "thank you i"), 26 frames,
two triangle strokes separated by a 3-frame rest gap. Offline path (`SeedPoseEngine` + numpy centroid +
lexicon).

| Step | State | Observed signal | vs threshold | Branch | Note |
|---|---|---|---|---|---|
| 0 | ingest | `n_frames = 26`, kind=`seed` | `>= 8` ✔ | proceed | 26 frames, routed as pose |
| 1 | segment | rest run at frames 11–13 (`v ≤ q30`) | run `3 ≥ 2` ✔ | 2 segments | spans [0–10], [14–25] |
| 2 | recognize | seg0 `THANK-YOU` conf 0.93; seg1 `ME` conf 0.88 | both `≥ 0.15` ✔ | confident | both in lexicon, no OOV |
| 3 | translate | text "thank you i"; round-trip chrF 0.81 | `≥ 0.20` ✔ | keep | verification passed |
| 4 | finalize | bad ratio `0/2 = 0.0` | `≤ 0.50` ✔ | **answer** | glosses + text + per-sign conf returned |

**Result:**
```json
{
  "decision": "answer",
  "glosses": ["THANK-YOU", "ME"],
  "text": "thank you i",
  "per_sign_confidence": [0.93, 0.88],
  "needs_review": false
}
```

**Contrast — pure-noise input** (random keypoints, no structure):

| Step | State | Observed signal | vs threshold | Branch |
|---|---|---|---|---|
| 0 | ingest | `n_frames = 30` | `>= 8` ✔ | proceed |
| 1 | segment | no rest run found | — | single span (fallback) |
| 2 | recognize | gloss conf 0.04 | `< 0.15` ✘ | low-conf |
| 3 | translate | text produced; round-trip chrF 0.05 | `< 0.20` ✘ | verify_failed |
| 4 | finalize | bad ratio `1/1 = 1.0` | `> 0.50` ✘ | **ABSTAIN** |

**Result:**
```json
{
  "decision": "uncertain",
  "glosses": [],
  "text": null,
  "needs_review": true,
  "note": "Low-confidence / unintelligible signing — please repeat more slowly."
}
```

This contrast *is* the value-add: an end-to-end decoder would have emitted a fluent, confident sentence for
the noise input. The agent recognizes that it has no trustworthy evidence and **declines**.

---

## 10. Where it lives & what it reuses

- **Module:** `src/signlang/agent/` — `config.py` (`AgentConfig`), `trace.py` (`ToolTrace`), `fsm.py` (the
  five states), `brain.py` (optional advisory LLM).
- **Reuse:** the P13/P14 MT chrF implementation (D4 verification), the embedded-gold Seed/Stub offline pattern
  from P15/P17/P19/P20 (the deterministic, network-free path), and the standard config/logging/trace
  scaffolding.
- **New for P01:** the motion-based **segmenter** (D2), the per-segment confidence **gate** (D3), the
  round-trip translation **verifier** (D4), and the OOV/low-confidence **abstention** policy (D5) — a sign-
  translation agent whose explicit job is to *know when not to translate*.
