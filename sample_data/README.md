# Sample data — Sign Language Translation

This system needs **no downloaded data to run**: the synthetic pose-sequence generator
(`signlang.data.synth_pose`) and the gloss→text lexicon (`signlang.data.lexicon`) produce signed
sentences (deterministic keypoint trajectories) + gold glosses/text, and the `SeedPoseEngine` reads
the gold embedded in each sequence — so the whole pipeline (segment → recognize → translate → eval →
agent) runs offline with no MediaPipe / torch / video.

- [`sample_sentences.json`](sample_sentences.json) — example seeds with their gold gloss/text (an
  in-vocabulary sentence and a note on the abstain behaviour for noisy/unclear input).

Render a synthetic collection to disk (`.npz` keypoint arrays + gold `.json`):

```bash
signlang gen-synthetic        # -> $SIGNLANG_DATA_DIR/synthetic/eval/
```

Translate one by seed:

```bash
signlang translate --seed 5000 --fast
```

To use a **real** corpus, set `data.use_hf: true`. Only `Sigurdur/icelandic-sign-language`
(Apache-2.0) is cleanly permissive and loaded as a smoke test; every continuous-SLT corpus
(`Exploration-Lab/iSign`, How2Sign, WLASL, …) is non-commercial / gated / unspecified — see
[`docs/data_card.md`](../docs/data_card.md). The synthetic generator is the primary data.
