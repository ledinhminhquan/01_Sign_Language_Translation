# Colab / H100 Guide — Sign Language Translation

End-to-end: upload to Drive → open the notebook → run cells → train → collect `report.pdf` +
`slides.pptx` + the submission bundle. The notebook auto-adapts to **H100 / A100 / L4 / T4**.

## Option A — clone from GitHub (recommended)

1. Push this folder to a GitHub repo (`01_Sign_Language_Translation`).
2. Open `notebooks/Sign_Language_Translation_Colab_H100.ipynb` in Colab
   (or upload it). **Runtime → Change runtime type → GPU** (pick H100 with Colab Pro+).
3. In the **Controls** cell, set `GIT_URL` to your repo. Leave `CLONE_FROM_GIT = True`.
4. **Runtime → Run all.** The **ONE-BUTTON AUTOPILOT** cell does everything.

## Option B — upload the repo to Drive

1. Zip this folder and upload to `MyDrive/signlang/`, unzip so that
   `MyDrive/signlang/01_Sign_Language_Translation/src/signlang/` exists.
2. In **Controls** set `CLONE_FROM_GIT = False`. Run all.

## What the autopilot produces (in your Drive)

```
MyDrive/signlang/artifacts/
  models/<version>/                 trained pose→gloss recognizer (+ translator/)
  runs/<run>/eval.json              gloss WER/acc + BLEU/chrF vs baselines + segmentation + abstain
  runs/<run>/{error_analysis,quality,benchmark,tune,monitoring}/
  submission/submission-*/report.pdf
  submission/submission-*/slides.pptx
  submission/submission-*/submission_bundle.zip   <-- hand this in
```

## Controls

| Control | Meaning |
|---------|---------|
| `TRAIN_CORE` | train the neural pose→gloss recognizer (off → numpy centroid, still produces a report) |
| `TRAIN_TRANSLATOR` | fine-tune t5-small gloss→text (off → lexicon translator) |
| `N_SENTENCES` | number of synthetic signed sentences to generate for training |
| `RUN_AUTOPILOT` | one-button everything vs the individual step cells |

## GPU auto-profile

| GPU | batch size | bf16 / tf32 |
|-----|-----------|-------------|
| H100 | 192 | yes |
| A100 | 128 | yes |
| L4 | 64 | yes |
| T4 | 32 | no |
| CPU | 16 | no (training skipped by autopilot) |

## Testing after training

Cell **10** loads the trained model and translates seeds 5000–7777 (showing glosses → text vs gold).
Cell **9** runs `grade` (target score **1.0**) and `demo-agent`.

## Notes

- **Colab-safe install:** the ML deps come from `requirements_colab.txt`, then the package installs
  with `pip install -e . --no-deps` so it never perturbs Colab's torch/CUDA.
- **Offline-friendly:** with no GPU the autopilot skips training and runs the SeedPose + numpy
  centroid + lexicon path end-to-end (still writes `report.pdf`, `slides.pptx`, and grades).
- **Real video:** install `mediapipe` (in `requirements_colab.txt`) and pass real pose via the API's
  `frames` field, or extend `data/dataset.py` to load a corpus. Every continuous-SLT corpus is
  non-commercial / gated (see `docs/data_card.md`) — the synthetic generator avoids that.
- **Ethics:** sign-language video is biometric. This is assistive tooling, not a replacement for a
  human interpreter; do not deploy on real signers without consent and Deaf-community involvement.
