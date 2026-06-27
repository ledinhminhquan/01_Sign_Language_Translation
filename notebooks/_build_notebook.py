"""Generate the H100/Colab training notebook (writes a valid .ipynb JSON).

Run:  python notebooks/_build_notebook.py
Produces: notebooks/Sign_Language_Translation_Colab_H100.ipynb

The notebook: controls (#@param) -> GPU check -> Drive mount + env paths -> git clone/upload ->
Colab-safe install (requirements_colab.txt, then `pip install -e . --no-deps`) -> GPU auto-profile
-> write train_colab.yaml -> gen-synthetic -> ONE-BUTTON autopilot -> individual steps ->
diagnostics -> test the trained model -> locate deliverables. Auto-adapts H100/A100/L4/T4.
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "Sign_Language_Translation_Colab_H100.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(
    "# Sign Language Translation — Colab / H100 Training\n"
    "\n"
    "**Translate sign-language video / pose-keypoint sequences into spoken text** via a gloss "
    "intermediate (Sign2Gloss2Text). A frozen pose front-end (MediaPipe Holistic on Colab, a "
    "SeedPoseEngine offline) feeds a **trainable seq2seq core** (a transformer pose→gloss recognizer "
    "+ a t5-small gloss→text translator). A deterministic agent segments the signs, gates recognition "
    "confidence, verifies the translation, and **abstains** on out-of-vocabulary signing.\n"
    "\n"
    "**One button:** run the *Setup* cells, then **ONE-BUTTON AUTOPILOT**. It auto-detects the GPU "
    "(H100 → A100 → L4 → T4), trains the core, evaluates vs baselines, runs analysis, and writes "
    "`report.pdf` + `slides.pptx` + a submission bundle to your Drive.\n"
    "\n"
    "_Author: Le Dinh Minh Quan (23127460) — NLP in Industry, Final Assignment (P01, the last project)._\n"
    "\n"
    "> Note: every continuous sign-language corpus is non-commercial / gated (see `docs/data_card.md`), "
    "so the **synthetic pose generator is the primary data** — a defensible, honest design choice."))

CELLS.append(md("## 0. Controls"))
CELLS.append(code(
    "#@title Controls { run: 'auto' }\n"
    "USE_DRIVE = True           #@param {type:'boolean'}\n"
    "CLONE_FROM_GIT = True      #@param {type:'boolean'}\n"
    "GIT_URL = 'https://github.com/<your-username>/01_Sign_Language_Translation.git'  #@param {type:'string'}\n"
    "TRAIN_CORE = True          #@param {type:'boolean'}\n"
    "TRAIN_TRANSLATOR = True    #@param {type:'boolean'}   # fine-tune t5 gloss->text (else lexicon)\n"
    "N_SENTENCES = 1200         #@param {type:'integer'}\n"
    "RUN_AUTOPILOT = True       #@param {type:'boolean'}\n"
    "print('controls set')"))

CELLS.append(md("## 1. GPU check"))
CELLS.append(code(
    "!nvidia-smi -L || echo 'No GPU — runtime > Change runtime type > GPU (H100/A100/L4/T4)'\n"
    "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),\n"
    "                    '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"))

CELLS.append(md("## 2. Drive mount + paths"))
CELLS.append(code(
    "import os\n"
    "if USE_DRIVE:\n"
    "    from google.colab import drive; drive.mount('/content/drive')\n"
    "    BASE = '/content/drive/MyDrive/signlang'\n"
    "else:\n"
    "    BASE = '/content/signlang'\n"
    "os.makedirs(BASE, exist_ok=True)\n"
    "os.environ['SIGNLANG_ARTIFACTS_DIR'] = BASE + '/artifacts'\n"
    "os.environ['HF_HOME'] = BASE + '/hf'\n"
    "print('artifacts ->', os.environ['SIGNLANG_ARTIFACTS_DIR'])"))

CELLS.append(md("## 3. Get the code"))
CELLS.append(code(
    "import os\n"
    "if CLONE_FROM_GIT:\n"
    "    if not os.path.isdir('/content/repo'):\n"
    "        !git clone $GIT_URL /content/repo\n"
    "    else:\n"
    "        !cd /content/repo && git pull --ff-only || true\n"
    "    PROJ = '/content/repo'\n"
    "else:\n"
    "    PROJ = BASE + '/01_Sign_Language_Translation'\n"
    "os.environ['PROJ'] = PROJ\n"
    "assert os.path.isdir(PROJ + '/src/signlang'), 'repo not found at ' + PROJ\n"
    "print('repo at', PROJ)"))

CELLS.append(md("## 4. Install (Colab-safe)\n"
                "Install the ML/serving/report deps from `requirements_colab.txt` (torch is "
                "preinstalled on Colab), then the package with `--no-deps` so it does not perturb "
                "Colab's resolved torch/CUDA."))
CELLS.append(code(
    "!pip -q install -r $PROJ/requirements_colab.txt\n"
    "!pip -q install -e $PROJ --no-deps\n"
    "import importlib, signlang; importlib.reload(signlang)\n"
    "print('signlang', signlang.__version__)"))

CELLS.append(md("## 5. GPU auto-profile → write `train_colab.yaml`\n"
                "Batch size auto-scales by GPU tier (H100 → A100 → L4 → T4)."))
CELLS.append(code(
    "import torch, yaml, os\n"
    "name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'\n"
    "if 'H100' in name: bs, bf16, tf32 = 192, True, True\n"
    "elif 'A100' in name: bs, bf16, tf32 = 128, True, True\n"
    "elif 'L4' in name:  bs, bf16, tf32 = 64, True, True\n"
    "elif 'T4' in name:  bs, bf16, tf32 = 32, False, False\n"
    "else:               bs, bf16, tf32 = 16, False, False\n"
    "cfg = {'data': {'use_hf': False, 'n_sentences': int(N_SENTENCES), 'vocab_size': 40, 'seed': 42},\n"
    "       'pose': {'engine': 'auto', 'normalize': True},\n"
    "       'model': {'arch': 'pose2seq_transformer', 'd_model': 256, 'enc_layers': 4,\n"
    "                 'text_backbone': 't5-small', 'use_gloss_intermediate': bool(TRAIN_TRANSLATOR),\n"
    "                 'num_train_epochs': 25, 'per_device_train_batch_size': bs, 'bf16': bf16, 'tf32': tf32},\n"
    "       'agent': {'llm_fallback_enabled': False}}\n"
    "os.makedirs(PROJ + '/configs', exist_ok=True)\n"
    "open(PROJ + '/configs/train_colab.yaml','w').write(yaml.safe_dump(cfg, sort_keys=False))\n"
    "print(f'GPU={name} -> batch_size={bs} bf16={bf16} tf32={tf32}')\n"
    "print(open(PROJ + '/configs/train_colab.yaml').read())"))

CELLS.append(md("## 6. Sanity-check + render a synthetic collection"))
CELLS.append(code(
    "!cd $PROJ && signlang --config configs/train_colab.yaml data\n"
    "!cd $PROJ && signlang --config configs/train_colab.yaml gen-synthetic"))

CELLS.append(md("## 7. ONE-BUTTON AUTOPILOT 🚀\n"
                "data → baseline → **train recognizer (+ t5 translator)** → evaluate → tune → "
                "error-analysis → quality → benchmark → demo → monitoring → **report.pdf + "
                "slides.pptx** → grade → zipped bundle. Each step is isolated; training is skipped "
                "automatically if no GPU is present."))
CELLS.append(code(
    "flag = '' if TRAIN_CORE else '--no-train'\n"
    "if RUN_AUTOPILOT:\n"
    "    !cd $PROJ && signlang --config configs/train_colab.yaml autopilot $flag\n"
    "else:\n"
    "    print('RUN_AUTOPILOT is off — use the individual steps below.')"))

CELLS.append(md("## 8. Individual steps (optional)"))
CELLS.append(code(
    "# Train only (recognizer + t5 translator):\n"
    "# !cd $PROJ && signlang --config configs/train_colab.yaml train\n"
    "# Evaluate (full, with the trained core):\n"
    "# !cd $PROJ && signlang --config configs/train_colab.yaml evaluate\n"
    "# Pose-noise robustness sweep:\n"
    "# !cd $PROJ && signlang --config configs/train_colab.yaml tune\n"
    "# Report + slides only:\n"
    "# !cd $PROJ && signlang --config configs/train_colab.yaml generate-report\n"
    "# !cd $PROJ && signlang --config configs/train_colab.yaml generate-slides"))

CELLS.append(md("## 9. Diagnostics"))
CELLS.append(code(
    "!cd $PROJ && signlang --config configs/train_colab.yaml grade\n"
    "!cd $PROJ && signlang --config configs/train_colab.yaml demo-agent"))

CELLS.append(md("## 10. Test the trained model"))
CELLS.append(code(
    "from signlang.config import load_config\n"
    "from signlang.agent.translate_agent import TranslationAgent\n"
    "from signlang.data.synth_pose import make_sentence\n"
    "cfg = load_config(PROJ + '/configs/train_colab.yaml')\n"
    "agent = TranslationAgent(cfg, load_model=True)  # loads the trained recognizer/translator if present\n"
    "for s in [5000, 5001, 5002, 7777]:\n"
    "    seq = make_sentence(s, cfg)\n"
    "    out = agent.translate(seq)\n"
    "    print(s, '->', out['glosses'], '=>', repr(out['text']), '| gold:', repr(seq.spec['text']),\n"
    "          '| abstained=', out['abstained'])"))

CELLS.append(md("## 11. Locate deliverables"))
CELLS.append(code(
    "import glob, os\n"
    "root = os.environ['SIGNLANG_ARTIFACTS_DIR']\n"
    "for pat in ['submission/*/report.pdf','submission/*/slides.pptx','submission/*/submission_bundle.zip',\n"
    "            'runs/*/eval.json','models/*']:\n"
    "    for p in glob.glob(os.path.join(root, pat)):\n"
    "        print(p)\n"
    "print('\\nDownload report.pdf + slides.pptx + submission_bundle.zip from the path above (in your Drive).')"))


def main():
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "toc_visible": True},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Wrote", NB)


if __name__ == "__main__":
    main()
