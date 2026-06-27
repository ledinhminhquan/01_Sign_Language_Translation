"""Hugging Face Space / standalone entrypoint for the Sign Language Translation demo.

Launches the Gradio UI: pick a signed sentence (seed) -> the agent extracts the pose, segments the
signs, recognizes the glosses, and translates to spoken text. Runs offline (SeedPoseEngine + numpy
centroid recognizer + lexicon translator) when torch/mediapipe are absent.
"""

from __future__ import annotations

import os

from signlang.api.ui import build_demo
from signlang.config import AppConfig

if __name__ == "__main__":
    cfg = AppConfig()
    load_model = os.environ.get("SIGNLANG_LOAD_MODEL", "1") not in ("0", "false", "False")
    demo = build_demo(cfg, load_model=load_model)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
