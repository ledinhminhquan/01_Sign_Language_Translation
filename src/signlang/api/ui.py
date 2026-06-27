"""Gradio demo UI for sign-language translation.

Pick a seed (or random) -> a synthetic signed sentence is generated, the agent extracts the pose,
segments the signs, recognizes the glosses, and translates to spoken text - showing the recognized
gloss sequence, the text, per-sign confidence, and whether it abstained. Heavy deps lazy.
"""

from __future__ import annotations

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_demo(cfg: AppConfig = None, load_model: bool = True):
    import gradio as gr
    from ..agent.translate_agent import TranslationAgent
    from ..data.synth_pose import make_sentence

    cfg = cfg or AppConfig()
    agent = TranslationAgent(cfg, load_model=load_model)

    def run(seed):
        seq = make_sentence(int(seed), cfg)
        out = agent.translate(seq)
        gloss_line = "  ".join(f"{g}({c:.2f})" for g, c in zip(out["glosses"], out["gloss_confs"]))
        status = out["status"]
        if out["abstained"]:
            status += " — abstained (sign unclear / out-of-vocabulary; please repeat)"
        elif out["low_confidence"]:
            status += " — low confidence (needs review)"
        return gloss_line, out["text"], f"gold: {seq.spec['text']}", status

    with gr.Blocks(title=cfg.serving.api_title) as demo:
        gr.Markdown(f"# {cfg.serving.api_title}\n"
                    "Translate a **signed sentence** (pose-keypoint sequence) to spoken text. The agent "
                    "segments the signs, recognizes the **glosses**, translates to text, and **abstains** "
                    "when the signing is unclear. _(Synthetic signer for the demo; real input = MediaPipe "
                    "pose from video.)_")
        seed = gr.Slider(0, 9999, value=5000, step=1, label="Signed-sentence seed")
        btn = gr.Button("Translate", variant="primary")
        glosses = gr.Textbox(label="Recognized glosses (confidence)", lines=1)
        text = gr.Textbox(label="Spoken text", lines=1)
        gold = gr.Textbox(label="(reference)", lines=1)
        status = gr.Textbox(label="Status", lines=1)
        btn.click(run, inputs=[seed], outputs=[glosses, text, gold, status])
        gr.Markdown("_Assistive only — not a substitute for a human interpreter. Pose video is biometric data._")
    return demo


def launch(cfg: AppConfig = None, share: bool = False, **kwargs):
    build_demo(cfg).launch(share=share, **kwargs)


__all__ = ["build_demo", "launch"]
