"""signlang - Sign Language Translation.

Translate sign-language video / pose-keypoint sequences into spoken-language text (via an
intermediate gloss). A pretrained/algorithmic pose front-end (MediaPipe Holistic on Colab, a
SeedPoseEngine offline) feeds a trainable sign-to-gloss/text seq2seq core; a deterministic agent
segments the signs, gates recognition confidence, verifies the translation, and abstains on
out-of-vocabulary input. Runs fully offline on a synthetic pose-sequence generator.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Le Dinh Minh Quan"

__all__ = ["__version__", "__author__"]
