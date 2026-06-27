# P01 — Sign Language Translation · Design Brief

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment (the last & hardest project).
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`.

This brief locks the stack after verifying every model/dataset id on the Hugging Face Hub (authenticated as
`ledinhminhquan`). Headline finding: **there is NO permissively-licensed, directly-loadable Sign→Text/Gloss→Text
seq2seq checkpoint and NO permissive continuous-SLT corpus on the Hub** — every continuous benchmark is non-commercial,
gated, or unspecified. This *validates* the planned architecture: train a small seq2seq core from a permissive backbone,
driven by a **synthetic pose-sequence generator + stub recognizer** as the primary offline data (mirroring P15/P17/P19/P20).

---

## 1. Task & architecture

Sign Language Translation (SLT) = map a sign-language **video → spoken-language text**, optionally via an intermediate
**gloss** sequence (Camgöz et al.'s now-universal framing). We adopt the **Sign2Gloss2Text cascade** (best mirrors the
P13/P14/P15 cascades and gives a clean, measurable offline spine), with direct Sign2Text as a secondary mode.

```
VIDEO ──(MediaPipe Holistic, frozen)──► pose-keypoint sequence ──► segment into sign units
        ──► RECOGNIZE gloss per segment (the trainable core) ──► TRANSLATE gloss→text ──► assemble sentence
```

- **Front-end (PRETRAINED / ALGORITHMIC — NOT trained):** MediaPipe Holistic hand+body landmarks per frame on Colab;
  a **SeedPoseEngine** offline (reads the gold embedded in the synthetic generator). No license encumbrance (MediaPipe
  is an algorithmic Google lib). Permissive frozen-video alternative: `microsoft/xclip-base-patch32` (MIT).
- **Trainable core (the single measured unit):** a sign-segment → gloss recognizer + gloss → text seq2seq. Offline this
  is a **numpy nearest-centroid gloss classifier** (genuinely classifies the pose displacement; no torch) + the lexicon
  translator; on Colab it upgrades to a transformer recognizer + a `t5-small` translator.
- **Why a synthetic spine:** restrictive licensing is THE defining constraint of this field — so a reproducible
  synthetic pose generator that embeds the gold gloss/text is the primary data, and a defensible, honest design choice.

## 2. Verified model stack (HF Hub ids confirmed)

| Component | id | License | Role |
|---|---|---|---|
| **Trainable core (translator)** | **`google-t5/t5-small`** | **Apache-2.0** | PRIMARY; tiny (60.5M), fits T4; gloss→text seq2seq |
| Byte-level variant | `google/byt5-small` | Apache-2.0 | robust to OOV glosses / symbolic pose tokens |
| MT reuse (P13/P14) | `facebook/m2m100_418M` | MIT | multilingual gloss→text; reuses the existing MT harness |
| Pose→gloss reference | `manohonsy/how2sign-pose-cslr` | MIT | 4.8M pose+CTC CSLR on How2Sign — proof a ~5M pose model is student-scale (architecture ref) |
| Notation→text precedent | `sign/sockeye-signwriting-to-text` | MIT | "treat the recognized symbolic sequence as a source language, run standard MT" (Sockeye, not HF transformers) |
| Frozen video front-end (alt) | `microsoft/xclip-base-patch32` | MIT | permissive video encoder if going video→features instead of pose |

**Do NOT depend on any pretrained SLT checkpoint as the measured component** — none load cleanly into `transformers` as a
Sign→Text translator. **AVOID (non-commercial):** `MCG-NJU/videomae-base` (CC-BY-NC), `sign/mediapipe-vq` (CC-BY-NC-SA),
`sign/sockeye-text-to-factored-signwriting` (CC-BY-NC). **Unspecified = all-rights-reserved:** `sign/signwriting-clip`,
the `PhoenixHu/grpo_internvl2_5_how2sign_*` fine-tunes.

## 3. Verified dataset stack (all flagged where restrictive)

| id | What | License | Flag |
|---|---|---|---|
| **synthetic generator** | deterministic pose trajectories + embedded gold gloss/text | n/a (ours) | **PRIMARY — non-negotiable** |
| `Sigurdur/icelandic-sign-language` | YouTube-SL-25 Icelandic slice (214 rows; `video_id`, timed `transcript`) | **Apache-2.0** ✅ | the ONLY cleanly permissive real corpus → real-data **smoke test** |
| `om192006/sign_language_keypoints` | pre-extracted MediaPipe keypoints, 29 gestures | **MIT** ✅ | tiny/isolated; pose-schema template |
| `PSewmuthu/How2Sign_Holistic` | MediaPipe Holistic landmark sequences (.npy) for How2Sign | MIT tag | **FLAG: derived from How2Sign (NC upstream)** |
| `aipieces/How2Sign` | How2Sign ASL→English video + keypoints | unspecified | **FLAG: upstream How2Sign is CC-BY-NC** |
| `Exploration-Lab/iSign` | Indian SL; defines **SignPose2Text** (this exact task), 118K+ pairs | CC-BY-NC-SA + **GATED** | **FLAG: NC + ShareAlike + gated** |
| `Kibalama/poseformer-sign-language` | WLASL → MediaPipe landmarks | unset (WLASL "other") | **FLAG: isolated, non-permissive source** |
| `Voxel51/WLASL` | 11,980 isolated-sign videos | other | **FLAG: research terms; recognition only** |

**NOT on the Hub (do not invent ids):** RWTH-PHOENIX-Weather-2014T, CSL-Daily, YouTube-ASL/SL-25 as clean redistributable
repos — they live on university servers under academic licenses. `Exploration-Lab/ISLTranslate` does not exist (use `iSign`).

## 4. The offline synthetic spine (mirrors P15/P17/P19/P20)

`data/synth_pose.py`: for each gloss in a small vocabulary (`data/lexicon.py`, 40 ASL-style glosses with a gloss→text
map) fix a deterministic **motion direction** in keypoint space (seeded by gloss index). A sign = a smooth up-down stroke
along that direction (so the per-sign mean displacement recovers the gloss); signs are separated by near-still **rest
frames** (so a motion-based segmenter splits them). A sentence concatenates 2–6 signs; the spoken text comes from the
lexicon (deliberately ≠ the gloss tokens — e.g. `THANK-YOU`→"thank you", `ME`→"i" — so the translation stage is
non-trivial vs identity). The gold spec `{glosses, text, boundaries}` is **embedded on the sequence**; the
`SeedPoseEngine`/`SeedRecognizer` read it back. → the whole pipeline (segment → recognize → translate → eval → agent →
tests) runs with **NO mediapipe / torch / video / network**.

The keypoint layout (`pose/layout.py`) mirrors real data: 2×21 hand + 25 body landmarks × 3 coords = a per-frame vector,
the same shape MediaPipe Holistic / `PSewmuthu/How2Sign_Holistic` produce.

## 5. Metrics & baselines

**Translation:** `BLEU-1..4` (**BLEU-4 = headline**), `ROUGE-L`, **chrF/chrF++** (reuse P13/P14). **Recognition (CSLR
stage):** **gloss-WER** (sub/del/ins over gloss tokens) + position-aligned gloss accuracy + sequence-exact-match. Isolated
recognition would use Top-1/5 accuracy. Reported on dev + test (here: held-out synthetic split).

> **Honesty caveat (cite):** Yazdani et al., "A Critical Study of Automatic Evaluation in SLT" (2025,
> hf.co/papers/2510.25434) — BLEU/chrF/ROUGE/BLEURT are unreliable for SLT (length-sensitive, blind to hallucination /
> semantic equivalence). We report the standard set **and flag these limitations** in the docs + autoreport.

**Baselines (isolate the trainable core):** (1) **identity/passthrough** (gloss tokens as text — shows the reordering the
model must learn); (2) **most-frequent gloss** recognizer; (3) **random gloss**; (4) **Seed oracle** (perfect recognition
→ upper bound on the translate stage). The trained core must beat 1–3 and approach 4.

## 6. The agent — 5 decision points (the mandatory agentic component)

Deterministic FSM, every step traced; optional LLM brain (`anthropic`) OFF by default (advisory only, never changes output).

| # | State | Decision (acts on) | Branches |
|---|-------|--------------------|----------|
| **D1** | ingest | input gate: frame count ≥ `min_frames`; route video→pose vs already-pose | proceed / fail |
| **D2** | segment | motion-based sign segmentation (low-velocity rest frames split signs) | n segments / single span |
| **D3** | recognize | per-segment gloss + confidence; below `recog_min_conf` → flag low-confidence | confident / low-conf |
| **D4** | translate+verify | gloss→text; optional round-trip back-translate (text→gloss agreement) / chrF keep-gate | keep / re-flag |
| **D5** | finalize | **abstain** if the OOV/low-confidence segment ratio > `oov_abstain_ratio` → "uncertain" + needs_review | answer / abstain |

**Value-add:** explicit gloss-confidence gating + translation verification + **abstention** on out-of-vocabulary /
unintelligible signing (beats a blind end-to-end decode that hallucinates fluent text from noise).

## 7. The 10 Section-I docs

problem_definition · data_description · data_card · model_selection · architecture · agent_architecture ·
translation_evaluation (the special quality doc — metrics + the reliability caveat) · deployment ·
continual_learning_monitoring · privacy_robustness · project_plan · ethics_statement · model_card · slide_deck_outline.

## 8. Reuse map

- **From P13/P14 (MT):** BLEU/chrF/WER metric implementations + the seq2seq train/eval pattern + m2m100/t5 backbone.
- **From P15/P17/P19/P20:** the embedded-gold synthetic-generator + Seed/Stub offline pattern; the standard
  config/logging/registry/autoreport/charts/monitoring/automation/grading/cli/api templates.
- **NEW for P01:** the pose-keypoint layout + front-end (MediaPipe / SeedPoseEngine), the synthetic **pose-sequence**
  generator (trajectories, not images), the motion-based **segmenter**, the per-segment **gloss recognizer**
  (numpy nearest-centroid offline + transformer on Colab), and the sign-translation agent.

## 9. Deployment & ethics

- **Deploy:** FastAPI (`POST /translate`: pose sequence / video → gloss + text + per-sign confidence + abstain flag) +
  Gradio demo + Docker (mediapipe + ffmpeg + libGL) + HF Space. Offline BM-free path (SeedPoseEngine + centroid) on CPU.
- **Ethics/Privacy:** sign-language video is **biometric + identifying** (face/hands, Deaf-community data) → consent,
  on-device/edge processing, no retention by default, the LLM brain off. **Representation bias** is acute: models trained
  on one sign language / signer set fail on others — never present a low-confidence translation as authoritative
  (abstain + show per-sign confidence + keep a human in the loop, especially in medical/legal settings). Engage the Deaf
  community; SLT is an accessibility aid, not a replacement for human interpreters.
