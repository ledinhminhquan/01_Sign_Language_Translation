# Deployment — Sign Language Translation (`signlang`)

Three ways to serve the system. The agent segments the signs, recognizes the **glosses**, translates
to spoken **text**, returns per-sign **confidence**, and **abstains** ("uncertain") on unclear /
out-of-vocabulary signing.

> ⚠️ **Assistive only.** This is not a substitute for a qualified human interpreter, and sign-language
> video is **biometric, identifying data** — front the API with auth, process on-device where possible,
> and do not retain pose/video by default.

## 1. Local (FastAPI + Gradio)

```bash
pip install -e ".[api,report]"
signlang serve --ui          # API at :8000, demo UI at :8000/ui
```

```bash
curl -s localhost:8000/healthz
# translate a generated synthetic signed sentence (demo):
curl -s -X POST localhost:8000/translate -H 'content-type: application/json' -d '{"seed": 5000}'
# translate a real pose-keypoint sequence (T x keypoint_dim):
curl -s -X POST localhost:8000/translate -H 'content-type: application/json' -d '{"frames": [[...], [...]]}'
```

Response (abridged):

```json
{
  "glosses": ["QUESTION", "YOU", "BAD"], "gloss_confs": [0.44, 0.45, 0.44],
  "text": "question you bad", "n_segments": 3, "mean_conf": 0.44,
  "low_confidence": false, "abstained": false, "needs_review": false,
  "status": "completed", "recognizer": "centroid"
}
```

## 2. Docker

```bash
docker compose up --build        # bundles ffmpeg + libGL for video/MediaPipe
```

## 3. Hugging Face Space (Gradio)

Point a Gradio Space at [`app/app.py`](../app/app.py). Set `SIGNLANG_LOAD_MODEL=0` to force the
offline SeedPose + numpy centroid path on CPU-only Spaces.

## Environment variables

| Var | Purpose |
|-----|---------|
| `SIGNLANG_ARTIFACTS_DIR` | root for models / runs / logs |
| `SIGNLANG_MODEL_DIR` | trained recognizer / translator location |
| `SIGNLANG_USE_HF` | `1` to probe the permissive real corpus (default synthetic) |
| `SIGNLANG_INFER_CONFIG` | path to a YAML config the server loads |
| `HF_HOME` | Hugging Face cache |
| `SIGNLANG_LLM_API_KEY` | only if the optional LLM brain is enabled (off by default) |

## Notes

- **CPU vs GPU:** the pose front-end + the numpy centroid recognizer + lexicon translator run on CPU;
  the neural transformer recognizer + t5 translator use a GPU when `torch` with CUDA is present, else
  fall back to the offline path.
- **Real input:** the MediaPipe pose extraction turns an uploaded video into the keypoint sequence the
  `/translate` `frames` field expects.
