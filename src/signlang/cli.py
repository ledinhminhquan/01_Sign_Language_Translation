"""Command-line interface - the single entrypoint for the signlang system.

    signlang <command> [options]

Commands: data, gen-synthetic, train, train-baseline, tune, evaluate, translate, demo-agent,
serve, benchmark, error-analysis, quality, monitor-log, generate-report, generate-slides,
autopilot, grade.

All console output is ASCII-only (Windows cp1252 safe); stdout stays pipeable JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)

TITLE = "Sign Language Translation System"
AUTHOR = "Le Dinh Minh Quan"


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def cmd_data(args):
    from .data.download_dataset import download_all
    print(json.dumps(download_all(_load(args), render_synthetic=args.render), indent=2, ensure_ascii=False))


def cmd_gen_synthetic(args):
    from .data.dataset import build_synthetic
    print(json.dumps(build_synthetic(_load(args), split=args.split), indent=2, ensure_ascii=False))


def cmd_train(args):
    from .training.train_sign2text import train_all
    print(json.dumps(train_all(_load(args), limit=args.limit,
                               train_translator_stage=not args.no_translator), indent=2))


def cmd_train_baseline(args):
    from .training.train_baseline import build_baseline
    print(json.dumps(build_baseline(_load(args), limit=args.limit), indent=2, ensure_ascii=False))


def cmd_tune(args):
    from .training.tune import tune
    print(json.dumps(tune(_load(args), load_model=not args.fast), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    rep = evaluate(_load(args), limit=args.limit, load_model=not args.fast, pose_noise=args.pose_noise)
    print(json.dumps(rep.get("headline", rep), indent=2, ensure_ascii=False))


def cmd_translate(args):
    from .agent.translate_agent import TranslationAgent
    from .data.synth_pose import make_sentence
    cfg = _load(args)
    agent = TranslationAgent(cfg, load_model=not args.fast)
    seq = make_sentence(args.seed, cfg)
    out = agent.translate(seq)
    out["gold_text"] = seq.spec["text"]
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_demo_agent(args):
    from .agent.translate_agent import TranslationAgent
    from .data import samples
    cfg = _load(args)
    agent = TranslationAgent(cfg, load_model=not args.fast)
    for ex in samples.seed_eval(cfg, n=12):
        out = agent.translate(ex["seq"])
        ok = "OK" if out["text"].strip() == ex["text"].strip() else "(diff)"
        print(f"[{out['status']:12s}] {out['glosses']} -> '{out['text']}' {ok} abst={out['abstained']}")


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["SIGNLANG_INFER_CONFIG"] = str(args.config)
    target = "signlang.api.app_combined:app" if args.ui else "signlang.api.main:app"
    uvicorn.run(target, host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args)), indent=2, ensure_ascii=False))


def cmd_quality(args):
    from .analysis.quality import quality_report
    print(json.dumps(quality_report(_load(args)), indent=2, ensure_ascii=False))


def cmd_monitor_log(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    print(json.dumps(run_autopilot(_load(args), title=args.title, author=args.author,
                                   train=not args.no_train, limit=args.limit), indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signlang", description=TITLE)
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="prefetch/sanity-check (backbone + real smoke corpus + seed)")
    sp.add_argument("--render", action="store_true", help="also render the synthetic collection")
    sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("gen-synthetic", help="render a synthetic pose-sequence collection")
    sp.add_argument("--split", default="eval"); sp.set_defaults(func=cmd_gen_synthetic)
    sp = sub.add_parser("train", help="train the sign->gloss recognizer (+ t5 gloss->text translator)")
    sp.add_argument("--limit", type=int, default=None); sp.add_argument("--no-translator", action="store_true")
    sp.set_defaults(func=cmd_train)
    sp = sub.add_parser("train-baseline", help="fit + persist the numpy centroid recognizer (no GPU)")
    sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train_baseline)
    sp = sub.add_parser("tune", help="pose-noise robustness sweep")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="gloss WER/acc + BLEU/chrF vs baselines + segmentation")
    sp.add_argument("--limit", type=int, default=None); sp.add_argument("--pose-noise", type=float, default=0.0)
    sp.add_argument("--fast", action="store_true", help="offline centroid (no model download)")
    sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("translate", help="translate a synthetic signed sentence (by seed)")
    sp.add_argument("--seed", type=int, default=5000); sp.add_argument("--fast", action="store_true")
    sp.set_defaults(func=cmd_translate)
    sp = sub.add_parser("demo-agent", help="run the agent on the held-out synthetic split")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server (+ --ui for the Gradio demo)")
    sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--ui", action="store_true"); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark of the agent")
    sp.add_argument("--n", type=int, default=10); sp.add_argument("--warmup", type=int, default=2)
    sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="exact/abstained/wrong buckets + worst examples")
    sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("quality", help="recognition + translation + segmentation quality report")
    sp.set_defaults(func=cmd_quality)
    sp = sub.add_parser("monitor-log", help="production monitoring report from the job log")
    sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor_log)
    sp = sub.add_parser("generate-report", help="generate the PDF report")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: train -> eval -> analysis -> report+slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check")
    sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
