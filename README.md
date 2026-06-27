# Sign Language Translation (`signlang`)

> **Translate sign-language video / pose-keypoint sequences into spoken text** via a **gloss**
> intermediate (Sign2Gloss2Text). A frozen pose front-end (MediaPipe Holistic / a SeedPoseEngine
> offline) feeds a **trainable seq2seq core**; a deterministic agent (D1–D5) **segments** the signs,
> **gates recognition confidence**, **verifies** the translation, and **abstains** on out-of-vocabulary
> signing. Runs fully offline on a synthetic pose generator.

NLP in Industry — Final Assignment, **Project 01** (the last & hardest). Author: **Le Dinh Minh Quan (23127460)**.

This is a **video/pose → text sequence** task, unlike the earlier text/OCR projects. Only the
**sign-to-gloss/text seq2seq core** is trained; the pose extractor is a pretrained/algorithmic
front-end. The whole pipeline runs **fully offline** (a synthetic pose-sequence generator that embeds
the gold gloss/text + a `SeedPoseEngine` + a pure-numpy nearest-centroid recognizer + a lexicon
translator) and upgrades to **MediaPipe + a trained transformer recognizer + a fine-tuned t5
translator** on Colab/H100.

> **Why synthetic-first?** Every continuous sign-language corpus on the HF Hub is **non-commercial,
> gated, or unspecified-license** (verified), and no permissive Sign→Text checkpoint loads cleanly.
> So a reproducible synthetic pose generator is the primary data — a defensible, honest design choice.

---

## What it does

```
TRAIN:   video ──(MediaPipe Holistic, frozen)──► pose-keypoint sequence
SEGMENT: motion-based split into sign units  (rest frames separate signs)
RECOGNIZE: per-segment transformer / centroid ──► gloss + confidence
TRANSLATE: gloss sequence ──(t5-small / lexicon)──► spoken text
AGENT:   ingest(D1) ─► segment(D2) ─► recognize(D3) ─► translate+verify(D4) ─► abstain(D5)
OUTPUT:  glosses + spoken text + per-sign confidence  (or "uncertain" / abstain)
```

**Why an agent over an end-to-end decode?** It **segments** the signs, **gates per-sign confidence**,
**verifies** the translation (round-trip), and **abstains** on unclear / out-of-vocabulary signing —
instead of hallucinating fluent text from noise.

## Quickstart (offline, no GPU / no MediaPipe)

```bash
pip install -e .                       # core deps only (numpy, pyyaml, pydantic)

signlang demo-agent --fast             # run the agent on the held-out synthetic split
signlang translate --seed 5000 --fast  # translate a generated signed sentence
signlang evaluate --fast               # gloss WER/acc + BLEU/chrF vs baselines + segmentation
signlang autopilot --no-train          # full pipeline -> report.pdf + slides.pptx + bundle
signlang grade                         # rubric self-check (target score 1.0)
```

Everything above runs with **no torch, no MediaPipe, no network**.

## Train on Colab / H100

Open [`notebooks/Sign_Language_Translation_Colab_H100.ipynb`](notebooks/Sign_Language_Translation_Colab_H100.ipynb),
set the GPU runtime, **Run all** → the **one-button autopilot** trains the core and writes `report.pdf`
+ `slides.pptx` + a submission bundle to your Drive. Full walkthrough:
[`notebooks/COLAB_GUIDE.md`](notebooks/COLAB_GUIDE.md).

```bash
pip install -e ".[all]"                # torch + transformers + mediapipe + serving + report
signlang train                         # train the pose->gloss recognizer (+ t5 gloss->text translator)
signlang evaluate                      # full eval with the trained core
signlang serve --ui                    # FastAPI at :8000 + Gradio demo at :8000/ui
```

## Model & data stack (verified on the HF Hub)

| Component | Default | License |
|-----------|---------|---------|
| Pose front-end | **MediaPipe Holistic** (algorithmic); `SeedPoseEngine` offline | Apache-2.0 |
| Frozen video alt | `microsoft/xclip-base-patch32` | MIT |
| **Recognizer (trained)** | transformer over pose segments (numpy centroid offline) | — |
| Pose/CSLR reference | `manohonsy/how2sign-pose-cslr` (4.8M, pose+CTC) | MIT |
| **Translator (trained)** | **`google-t5/t5-small`** (gloss→text) | **Apache-2.0** |
| Translator alts | `facebook/m2m100_418M` (reuse P13/P14); `google/byt5-small` | MIT / Apache |
| Real smoke corpus | `Sigurdur/icelandic-sign-language` | Apache-2.0 |

⚠️ **Every continuous sign-language corpus is non-commercial / gated / unspecified** — `Exploration-Lab/iSign`
(CC-BY-NC-SA, gated), How2Sign (CC-BY-NC), `Voxel51/WLASL` (other), and `microsoft`-avoid pairs
(VideoMAE / `sign/mediapipe-vq`, CC-BY-NC). All flagged in [`docs/data_card.md`](docs/data_card.md)
and [`LICENSE`](LICENSE). RWTH-PHOENIX-2014T / CSL-Daily / YouTube-ASL are **not** redistributable HF
repos (academic licenses). The **synthetic pose generator** is the primary, license-clean data.

## Metrics

- **Recognition (CSLR):** gloss **WER** (sub/del/ins) + position-aligned accuracy + sequence exact-match.
- **Translation:** **BLEU-1..4** (BLEU-4 headline) + **chrF** + WER (reuse P13/P14).
- **Segmentation:** boundary-F1 vs the gold sign boundaries. **Abstention:** abstain rate.
- **Baselines:** most-frequent gloss, random gloss, identity-translate (gloss tokens as text), Seed oracle.

> **Honesty caveat:** automatic SLT metrics (BLEU/chrF/ROUGE/BLEURT) are unreliable — length-sensitive,
> blind to hallucination / semantic equivalence (Yazdani et al. 2025). We report the standard set and flag this.

Offline-verified: clean synthetic → **gloss accuracy 1.0, BLEU 99+, segmentation-F1 1.0** vs
most-frequent floor 0.02 and identity-translate BLEU ~84 (the lexicon adds ~15 BLEU); pose noise
degrades recognition (the robustness sweep); pure-noise input **abstains**.

## The agent — five decision points

| # | State | Decision (acts on) | Branches |
|---|-------|--------------------|----------|
| **D1** | ingest | frame-count gate; route video→pose | proceed / fail |
| **D2** | segment | motion-based sign segmentation | n segments / single span |
| **D3** | recognize | per-segment gloss + confidence vs `recog_min_conf` | confident / low-conf |
| **D4** | translate | gloss→text + round-trip verification | kept / flagged |
| **D5** | finalize | **abstain** if the low-confidence-segment ratio > `oov_abstain_ratio` | answer / abstain |

An optional LLM **brain** (`anthropic`, **OFF by default**) only adds an advisory note — it never
changes the glosses or the translation, and the default runs with **zero paid API calls**.

## Repository layout

```
src/signlang/
  config.py  cli.py  logging_utils.py
  pose/          engine.py layout.py            # MediaPipe / SeedPoseEngine / Stub + keypoint layout
  data/          synth_pose.py lexicon.py samples.py dataset.py download_dataset.py
  models/        sign2text.py baseline.py neural_recognizer.py neural_translator.py model_registry.py
  segmentation/  segmenter.py                   # motion-based sign segmentation
  training/      train_sign2text.py train_baseline.py evaluate.py tune.py metrics.py
  agent/         translate_agent.py tools.py policy.py state.py llm_orchestrator.py
  api/           main.py schemas.py dependencies.py ui.py app_combined.py
  analysis/      error_analysis.py latency.py quality.py
  autoreport/    artifact_loader.py charts.py report_pdf.py slides_pptx.py
  monitoring/    drift_report.py
  automation/    autopilot.py
  grading/       checklist.py
docs/   (14 Section-I docs + DESIGN_BRIEF)   notebooks/   tests/   configs/   app/   deploy/   sample_data/
```

## Documentation

All Section-I deliverables live in [`docs/`](docs/): problem definition, data description &
[data card](docs/data_card.md), [model selection](docs/model_selection.md),
[architecture](docs/architecture.md), [agent architecture](docs/agent_architecture.md),
[translation evaluation](docs/translation_evaluation.md), [deployment](docs/deployment.md),
[continual learning & monitoring](docs/continual_learning_monitoring.md),
[privacy & robustness](docs/privacy_robustness.md), [project plan](docs/project_plan.md),
[ethics statement](docs/ethics_statement.md), [model card](docs/model_card.md), and the slide outline.

## Tests

```bash
pip install -e . pytest && pytest -q     # offline; no torch / mediapipe / network
```

## Ethics

Sign-language video is **biometric and identifying** (face/hands, Deaf-community data). This is
**assistive** tooling, **not** a substitute for a qualified human interpreter. The system surfaces
per-sign confidence and abstains on unclear signing; deploy only with consent, edge processing, no
retention by default, and **Deaf-community involvement**. See [`docs/ethics_statement.md`](docs/ethics_statement.md).

## License

Code: [MIT](LICENSE). Third-party models/datasets retain their own licenses (sign-language corpora
are largely non-commercial / gated — see the data card). This MIT license covers only the code here.
