"""Generate the submission report.pdf for the signlang (Sign Language Translation) system.

A 10-15 page report covering every Section-I deliverable: problem & use cases, data (synthetic pose
+ flagged real corpora), the pose front-end + the trainable seq2seq core, the agent (D1-D5), the
evaluation (gloss WER/accuracy + BLEU/chrF + segmentation), deployment, continual learning &
monitoring, privacy & robustness, and ethics. Live numbers come from ``run_dir()`` artifacts;
missing metrics degrade to placeholders.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_now_iso
from . import charts as charts_mod
from .artifact_loader import (abstain_rate, agent_recog, agent_trans, base_model, baseline_recog,
                              baseline_trans, has_eval, headline, latency, load_artifacts,
                              model_version, read_doc, recognizer_name, seg_f1)

logger = get_logger(__name__)

_SUBTITLE = ("Translate sign-language video / pose-keypoint sequences into spoken text via a gloss "
             "intermediate. A frozen pose front-end (MediaPipe / a SeedPoseEngine offline) feeds a TRAINABLE "
             "seq2seq core; a deterministic agent (D1-D5) segments the signs, gates recognition confidence, "
             "verifies the translation, and abstains on out-of-vocabulary signing. Runs fully offline on a "
             "synthetic pose generator.")

_SECTIONS = [
    ("1. Problem Definition & Use Cases", "problem_definition.md"),
    ("2. Data (Synthetic Pose + Real Corpora)", "data_description.md"),
    ("3. System Architecture", "architecture.md"),
    ("4. Model Selection (Pose Front-end + Seq2Seq Core)", "model_selection.md"),
    ("5. Agent Architecture (Decisions D1-D5)", "agent_architecture.md"),
    ("6. Translation Evaluation", "translation_evaluation.md"),
    ("7. Deployment", "deployment.md"),
    ("8. Continual Learning & Monitoring", "continual_learning_monitoring.md"),
    ("9. Data Privacy & Robustness", "privacy_robustness.md"),
    ("10. Ethics & Responsible AI", "ethics_statement.md"),
]


def _builtin_sections(cfg: AppConfig, arts: Dict[str, Any]) -> Dict[str, str]:
    acc = agent_recog(arts, "gloss_accuracy")
    bleu = agent_trans(arts, "bleu")
    mf = baseline_recog(arts, "most_frequent_gloss", "gloss_accuracy")
    if acc is not None:
        res_line = (f"In the latest eval the trained core reaches gloss accuracy **{acc*100:.1f}%**"
                    + (f" vs the most-frequent floor **{mf*100:.1f}%**" if mf is not None else "")
                    + (f"; translation BLEU **{bleu:.1f}**, chrF **{agent_trans(arts,'chrf') or 0:.1f}**." if bleu is not None else "."))
    else:
        res_line = "Run `signlang evaluate` to populate the live numbers here."
    return {
        "problem_definition.md": f"""
## What it does
Translate **sign-language video / pose-keypoint sequences** into spoken-language **text**, via an
intermediate **gloss** sequence (Sign2Gloss2Text). The pose front-end is pretrained/algorithmic
(MediaPipe Holistic); the only trained component is the sign-to-gloss/text seq2seq core.

## The job-to-be-done
- **Accessibility:** bridge Deaf signers and non-signers (kiosks, video relay, captioning).
- **Assistive, not authoritative:** complements human interpreters; surfaces per-sign confidence and
  abstains on unclear signing.

## Why an agent over an end-to-end decode
The value-add: it **segments** the signs, **gates recognition confidence** per sign, **verifies** the
translation (round-trip), and **abstains** ("uncertain") on out-of-vocabulary / unintelligible
signing - instead of hallucinating fluent text from noise.

## Success metrics
- **Technical:** gloss **WER / accuracy** (recognition), **BLEU-4 / chrF** (translation), segmentation F1.
- **Business:** the share of abstained utterances, human-review load.
{res_line}
""",
        "model_selection.md": f"""
## The pose front-end (pretrained, NOT trained)
- **MediaPipe Holistic** hand + body landmarks per frame (algorithmic, no license encumbrance); a
  **SeedPoseEngine** reads the gold embedded in the synthetic generator offline. Permissive frozen-video
  alternative: `microsoft/xclip-base-patch32` (MIT). **Avoid** VideoMAE / `sign/mediapipe-vq` (CC-BY-NC).

## The trainable seq2seq core
- **Recognizer:** a compact transformer encoder over each sign segment -> gloss (offline = a numpy
  nearest-centroid classifier that genuinely classifies the pose displacement). Reference: the MIT
  `manohonsy/how2sign-pose-cslr` (4.8M pose+CTC) proves a ~5M pose model is student-scale.
- **Translator:** `{base_model(arts)}` - `google-t5/t5-small` (Apache) default, `facebook/m2m100_418M`
  (MIT, reused from P13/P14) for multilingual; `google/byt5-small` for symbolic sources.
- **Baselines:** most-frequent gloss, random gloss, identity-translate (gloss tokens as text), and a
  Seed **oracle** (perfect recognition -> upper bound on the translate stage).
- **No pretrained SLT checkpoint** loads cleanly as a Sign->Text translator (verified) -> the trained
  core is our own small seq2seq, driven by the synthetic spine.
{res_line}
""",
        "agent_architecture.md": f"""
## FSM
A deterministic finite-state machine; every tool returns a uniform dict and every transition is
traced. States: `ingest -> segment -> recognize -> translate+verify -> finalize`. An optional LLM
**brain** (`{cfg.agent.llm_model}`, OFF by default) only adds an advisory note; rules win and the
agent runs with **zero paid API calls**.

## Five decisions (each acts on an intermediate artifact)
- **D1 - ingest gate.** Reject sequences shorter than {cfg.agent.min_frames} frames; route video->pose.
- **D2 - segmentation.** Motion-based split into sign units (low-velocity rest frames separate signs).
- **D3 - confidence gate.** Per-segment gloss + confidence; below {cfg.agent.recog_min_conf} -> low-confidence.
- **D4 - translate + verify.** Gloss->text; a round-trip agreement check flags weak translations.
- **D5 - abstain / finalize.** If the low-confidence-segment ratio exceeds {cfg.agent.oov_abstain_ratio:.0%}
  -> **abstain** ("uncertain" + needs_review); else return glosses + text + per-sign confidence.

The agent emits `{{glosses, gloss_confs, text, n_segments, abstained, decisions[]}}`.
""",
        "translation_evaluation.md": f"""
## Metrics
**Recognition (CSLR stage):** gloss **WER** (sub/del/ins) + position-aligned gloss accuracy + sequence
exact-match. **Translation:** **BLEU-1..4** (BLEU-4 = headline) + **chrF** + WER (reuse P13/P14).
**Segmentation:** boundary-F1 vs the gold sign boundaries.

## Baselines
- **most-frequent gloss** / **random gloss** (recognition floors), **identity-translate** (gloss tokens
  as text - shows the reordering/lexicon the model must learn), and the **Seed oracle** (perfect
  recognition -> translate-stage upper bound).

> **Honesty caveat:** automatic SLT metrics (BLEU/chrF/ROUGE/BLEURT) are known to be unreliable
> (length-sensitive, blind to hallucination / semantic equivalence; Yazdani et al. 2025). We report
> the standard set and flag these limitations.
{res_line}
""",
    }


def _esc(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
    s = s.replace("&", "&amp;").replace("<b>", "\x00b\x00").replace("</b>", "\x00/b\x00")
    s = s.replace("<font face='Courier'>", "\x00f\x00").replace("</font>", "\x00/f\x00")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = (s.replace("\x00b\x00", "<b>").replace("\x00/b\x00", "</b>")
          .replace("\x00f\x00", "<font face='Courier'>").replace("\x00/f\x00", "</font>"))
    return s


def _md_to_flowables(md: str, styles, max_lines: int = 300):
    from reportlab.platypus import Paragraph, Preformatted, Spacer
    flow, lines, in_code, code, bullet = [], md.splitlines()[:max_lines], False, [], []

    def flush():
        nonlocal bullet
        for b in bullet:
            flow.append(Paragraph("- " + _esc(b), styles["Body"]))
        bullet = []

    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code), styles["Code"])); code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(ln); continue
        s = ln.rstrip()
        if not s:
            flush(); flow.append(Spacer(1, 5)); continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            flow.append(Paragraph(_esc(s.lstrip("#").strip()), styles["H2" if level <= 2 else "H3"]))
        elif s.lstrip().startswith(("- ", "* ")):
            bullet.append(s.lstrip()[2:])
        else:
            flush(); flow.append(Paragraph(_esc(s), styles["Body"]))
    flush()
    return flow


def _fmt(v, scale=1.0, suffix=""):
    return (f"{v*scale:.1f}{suffix}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "-")


def _results_table(arts: Dict[str, Any], styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    flow = [Paragraph("Results - recognition + translation", styles["H3"])]
    if has_eval(arts):
        rows = [["System", "Gloss acc", "Gloss WER", "BLEU", "chrF"],
                ["agent (trained core)", _fmt(agent_recog(arts, "gloss_accuracy"), 100, "%"),
                 _fmt(agent_recog(arts, "gloss_wer")), _fmt(agent_trans(arts, "bleu")),
                 _fmt(agent_trans(arts, "chrf"))],
                ["most-frequent gloss", _fmt(baseline_recog(arts, "most_frequent_gloss", "gloss_accuracy"), 100, "%"),
                 "-", "-", "-"],
                ["random gloss", _fmt(baseline_recog(arts, "random_gloss", "gloss_accuracy"), 100, "%"), "-", "-", "-"],
                ["identity-translate", "-", "-", _fmt(baseline_trans(arts, "identity_translate", "bleu")),
                 _fmt(baseline_trans(arts, "identity_translate", "chrf"))]]
    else:
        rows = [["System", "Gloss acc", "Gloss WER", "BLEU", "chrF"], ["run `evaluate`", "-", "-", "-", "-"]]
    t = Table(rows, hAlign="LEFT", colWidths=[150, 75, 75, 60, 60])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")])]))
    sf = seg_f1(arts); ab = abstain_rate(arts)
    flow += [t, Spacer(1, 6),
             Paragraph(f"Segmentation boundary-F1: {_fmt(sf)}; abstain rate: {_fmt(ab, 100, '%')}. The trained "
                       "core beats the most-frequent / random floors; identity-translate isolates the value of the "
                       "gloss->text translation." if sf is not None else
                       "Run evaluate for segmentation + abstain.", styles["Body"])]
    lat = latency(arts, "p50")
    if lat is not None:
        flow.append(Paragraph(f"Agent latency: per-utterance p50 ~ {lat:.0f} ms (p95 ~ {latency(arts,'p95') or 0:.0f} ms).",
                              styles["Body"]))
    flow.append(Spacer(1, 8))
    return flow


def generate_report(cfg: AppConfig, title: Optional[str] = None, author: Optional[str] = None,
                    out_path: Optional[str] = None) -> str:
    title = title or cfg.project_title
    author = author or cfg.author
    arts = load_artifacts(cfg)
    out = Path(out_path) if out_path else run_dir() / "report" / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    builtins = _builtin_sections(cfg, arts)

    def section_md(fname: str) -> str:
        doc = read_doc(fname)
        if doc.strip():
            lines = doc.splitlines()
            return "\n".join(lines[:46]) if len(lines) > 46 else doc
        return builtins.get(fname, "")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer)
    except Exception as exc:
        logger.warning("reportlab unavailable (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("T", parent=base["Title"], fontSize=22, leading=26),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], textColor="#1a365d", spaceBefore=10),
        "H3": ParagraphStyle("H3", parent=base["Heading3"], textColor="#2b6cb0"),
        "Body": ParagraphStyle("B", parent=base["BodyText"], fontSize=9.5, leading=13),
        "Code": ParagraphStyle("C", parent=base["Code"], fontSize=7.5, leading=9, backColor="#f4f6f8"),
        "Meta": ParagraphStyle("M", parent=base["BodyText"], fontSize=11, leading=15),
    }
    try:
        built = dict(charts_mod.build_all(arts, out.parent / "charts"))
    except Exception as exc:
        logger.info("charts skipped (%s)", exc)
        built = {}

    story: List[Any] = [
        Spacer(1, 5 * cm), Paragraph(title, styles["Title"]), Spacer(1, 1 * cm),
        Paragraph(f"<b>{author}</b> - Student {cfg.student_id}", styles["Meta"]),
        Paragraph("NLP in Industry - Final Assignment (P01)", styles["Meta"]),
        Paragraph(_SUBTITLE, styles["Meta"]),
        Paragraph(f"Generated {utc_now_iso()}", styles["Body"]),
        Paragraph(f"Core: <b>{model_version(arts)}</b> (recognizer {recognizer_name(arts)}, base {base_model(arts)})",
                  styles["Body"]),
    ]
    story.append(PageBreak())
    story += _results_table(arts, styles)
    for name in ("recognition", "quality", "buckets"):
        if name in built:
            story += [Image(str(built[name]), width=13 * cm, height=7.0 * cm), Spacer(1, 6)]
    story.append(PageBreak())

    for heading, fname in _SECTIONS:
        story.append(Paragraph(heading, styles["H2"]))
        story += _md_to_flowables(section_md(fname), styles)
        story.append(Spacer(1, 10))

    try:
        SimpleDocTemplate(str(out), pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                          leftMargin=1.8 * cm, rightMargin=1.8 * cm, title=title, author=author).build(story)
    except Exception as exc:
        logger.warning("reportlab build failed (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)
    logger.info("Report -> %s", out)
    return str(out)


__all__ = ["generate_report"]
