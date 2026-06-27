"""Offline tests for the signlang pipeline: synth pose, segmenter, recognizer, metrics, agent, eval."""

from __future__ import annotations

import numpy as np
import pytest


# ---- synthetic pose generator -----------------------------------------------------------------

def test_make_sentence_deterministic(cfg):
    from signlang.data.synth_pose import make_sentence
    a = make_sentence(7, cfg)
    b = make_sentence(7, cfg)
    assert a.spec["glosses"] == b.spec["glosses"]
    assert a.spec["text"] == b.spec["text"]
    assert a.frames.shape[1] == cfg.pose.keypoint_dim
    assert len(a.spec["boundaries"]) == len(a.spec["glosses"])


def test_lexicon_translate(cfg):
    from signlang.data import lexicon
    txt = lexicon.translate_glosses(["THANK-YOU", "YOU"], cfg.data.vocab_size)
    assert txt == "thank you you"


# ---- pose engine + segmentation ----------------------------------------------------------------

def test_seed_pose_engine_passthrough(cfg):
    from signlang.data.synth_pose import make_sentence
    from signlang.pose.engine import extract_pose
    seq = make_sentence(3, cfg)
    out = extract_pose(seq, cfg.pose)
    assert out.spec["glosses"] == seq.spec["glosses"]
    assert out.frames.shape[0] == seq.frames.shape[0]


def test_segmenter_counts_signs(cfg):
    from signlang.data.synth_pose import make_sentence
    from signlang.pose.engine import extract_pose
    from signlang.segmentation.segmenter import segment, boundary_f1
    seq = extract_pose(make_sentence(11, cfg), cfg.pose)
    spans = segment(seq, cfg.agent)
    bf = boundary_f1(spans, make_sentence(11, cfg).spec["boundaries"])
    assert bf["f1"] >= 0.8


# ---- recognizer --------------------------------------------------------------------------------

def test_centroid_recognizer_accurate(cfg):
    from signlang.data import samples
    from signlang.models.sign2text import fit_centroid_recognizer
    from signlang.pose.engine import extract_pose
    rec = fit_centroid_recognizer(cfg, n_train=120)
    ev = samples.seed_eval(cfg, n=30)
    correct = total = 0
    for ex in ev:
        seq = extract_pose(ex["seq"], cfg.pose)
        preds = rec.recognize(seq, ex["boundaries"])
        for (g, _), gold in zip(preds, ex["glosses"]):
            total += 1
            correct += int(g == gold)
    assert correct / total > 0.9


# ---- metrics -----------------------------------------------------------------------------------

def test_translation_metrics():
    from signlang.training import metrics as M
    perfect = M.translation_metrics(["hello you"], ["hello you"])
    assert perfect["bleu"] == 100.0 and perfect["wer"] == 0.0
    assert M.chrf("abc", "abc") == 100.0


def test_recognition_metrics():
    from signlang.training import metrics as M
    rm = M.recognition_metrics([["A", "B"]], [["A", "C"]])
    assert rm["gloss_accuracy"] == 0.5
    assert rm["gloss_wer"] == 0.5


# ---- agent (D1-D5) -----------------------------------------------------------------------------

def test_agent_all_five_decisions(cfg):
    from signlang.agent.translate_agent import TranslationAgent
    from signlang.data import samples
    agent = TranslationAgent(cfg, load_model=False)
    job = agent.run(samples.seed_eval(cfg, n=1)[0]["seq"], save=False)
    assert [d.id for d in job.decisions] == ["D1", "D2", "D3", "D4", "D5"]


def test_agent_translates_correctly(cfg):
    from signlang.agent.translate_agent import TranslationAgent
    from signlang.data import samples
    agent = TranslationAgent(cfg, load_model=False)
    ev = samples.seed_eval(cfg, n=20)
    exact = sum(agent.translate(ex["seq"])["text"].strip() == ex["text"].strip() for ex in ev)
    assert exact / len(ev) > 0.8


def test_agent_short_sequence_fails(cfg):
    from signlang.agent.translate_agent import TranslationAgent
    from signlang.agent.state import JobStatus
    from signlang.data.synth_pose import PoseSequence
    agent = TranslationAgent(cfg, load_model=False)
    short = PoseSequence(frames=np.zeros((3, cfg.pose.keypoint_dim), dtype="float32"), fps=25, spec=None)
    job = agent.run(short, save=False)
    assert job.status is JobStatus.FAILED


def test_agent_abstains_on_noise(cfg):
    from signlang.agent.translate_agent import TranslationAgent
    from signlang.data.synth_pose import PoseSequence
    agent = TranslationAgent(cfg, load_model=False)
    rng = np.random.RandomState(0)
    noise = PoseSequence(frames=rng.normal(scale=0.05, size=(60, cfg.pose.keypoint_dim)).astype("float32"),
                         fps=25, spec=None)
    out = agent.translate(noise)
    assert out["abstained"] is True


# ---- evaluate / report / grading ---------------------------------------------------------------

def test_evaluate_offline_beats_baseline(cfg):
    from signlang.training.evaluate import evaluate
    rep = evaluate(cfg, save=False, load_model=False, limit=40)
    agent_acc = rep["systems"]["agent"]["recognition"]["gloss_accuracy"]
    mf_acc = rep["systems"]["most_frequent_gloss"]["recognition"]["gloss_accuracy"]
    assert agent_acc > mf_acc
    assert rep["systems"]["agent"]["translation"]["bleu"] > rep["systems"]["identity_translate"]["translation"]["bleu"] - 1


def test_evaluate_pose_noise_degrades(cfg):
    from signlang.training.evaluate import evaluate
    clean = evaluate(cfg, save=False, load_model=False, limit=40, pose_noise=0.0)
    noisy = evaluate(cfg, save=False, load_model=False, limit=40, pose_noise=0.6)
    assert clean["headline"]["gloss_accuracy"] >= noisy["headline"]["gloss_accuracy"]


def test_quality_report(cfg):
    from signlang.analysis.quality import quality_report
    r = quality_report(cfg, save=False)
    assert "gloss_accuracy" in r and "bleu" in r


def test_report_and_slides(cfg):
    from signlang.autoreport.report_pdf import generate_report
    from signlang.autoreport.slides_pptx import generate_slides
    rp = generate_report(cfg)
    sp = generate_slides(cfg)
    assert rp.endswith((".pdf", ".md"))
    assert sp.endswith((".pptx", ".md"))


def test_grading_runs():
    from pathlib import Path
    from signlang.grading.checklist import build_checklist
    repo = Path(__file__).resolve().parents[1]
    res = build_checklist(repo)
    assert res["summary"]["FAIL"] == 0, [i for i in res["items"] if i["status"] == "FAIL"]
