# P01 — Sign Language Translation · Deployment

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment.
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`.

This document describes how the Sign2Gloss2Text cascade is served: the FastAPI surface, the Gradio
demo, the Docker image (with the `mediapipe`/`ffmpeg`/`libGL` system dependencies the video→pose step
needs), the Hugging Face Space, the CPU-vs-GPU split across the pipeline stages, latency budgets,
environment-variable configuration, a concrete request/response pair, and the biometric-privacy notes
that are non-negotiable for sign-language video.

The deployment ships the **same agent FSM** described in `agent_architecture.md` — the five decision
points (ingest → segment → recognize → translate+verify → finalize/abstain) run inside the request
handler, so the served system inherits the per-sign confidence gating and the abstention behaviour
rather than blindly decoding fluent text from noise.

---

## 1. Serving topology at a glance

```
                         ┌──────────────────────────────────────────────────────────┐
 client ──video/pose──►  │  FastAPI (uvicorn)                                         │
                         │   /healthz   GET    liveness/readiness                     │
                         │   /vocabulary GET   the 40-gloss lexicon + gloss→text map  │
                         │   /translate  POST  seed | frames  →  agent FSM            │
                         │                                                            │
                         │   D1 ingest ──► D2 segment ──► D3 recognize ──► D4 verify  │
                         │                                       │                    │
                         │   front-end:  SeedPoseEngine (offline / CPU)               │
                         │            or MediaPipe Holistic (video→pose / CPU)        │
                         │   core:    numpy nearest-centroid (CPU)                    │
                         │            or transformer recognizer + t5-small (GPU)      │
                         └──────────────────────────────────────────────────────────┘
        Gradio demo (pick a signed sentence → glosses + text) mounts the same handler.
```

Two interchangeable serving profiles, selected entirely by config (no code change):

| Profile | Front-end | Recognizer | Translator | Hardware | Deps |
|---|---|---|---|---|---|
| **Offline / Seed** (default) | `SeedPoseEngine` | numpy nearest-centroid | lexicon map | **CPU only** | none beyond numpy |
| **Full / Colab-trained** | MediaPipe Holistic | transformer encoder | `google-t5/t5-small` | CPU front-end + **GPU** core | mediapipe, torch, transformers, ffmpeg, libGL |

The **default container is the offline profile** — it needs no torch, no mediapipe, no GPU, and no
network, so it is the path that actually runs in CI, in the autograder, and in the demo Space. The
full profile is opt-in via environment variables (§7).

---

## 2. FastAPI endpoints

The API lives in `src/signlang/api/` (built from the shared P13–P20 FastAPI template, extended for
pose payloads). Run it with:

```bash
uvicorn signlang.api.app:app --host 0.0.0.0 --port 8000
```

### 2.1 `GET /healthz`

Liveness/readiness probe. Returns the active serving profile and which components actually loaded, so
an orchestrator can tell a CPU-only Seed pod (always ready) from a GPU pod still warming the
transformer + t5 weights.

```json
{
  "status": "ok",
  "profile": "seed",
  "components": {
    "pose_frontend": "SeedPoseEngine",
    "recognizer": "numpy-nearest-centroid",
    "translator": "lexicon-map",
    "mediapipe_available": false,
    "torch_available": false
  },
  "vocab_size": 40,
  "version": "0.1.0"
}
```

`status` is `ok` once the lexicon, centroids, and (if enabled) model weights are loaded; readiness in
the full profile flips to `ok` only after the t5 and recognizer weights are resident on the device.

### 2.2 `GET /vocabulary`

Returns the closed 40-gloss vocabulary and the gloss→text lexicon map (`data/lexicon.py`). This is
the contract the client needs to interpret confidences and to understand abstention: a sign outside
this set is, by construction, out-of-vocabulary and is what drives the D5 abstain path.

```json
{
  "vocab_size": 40,
  "glosses": ["ME", "YOU", "THANK-YOU", "HELLO", "NAME", "..."],
  "lexicon": {
    "ME": "i",
    "YOU": "you",
    "THANK-YOU": "thank you",
    "HELLO": "hello"
  },
  "note": "Closed vocabulary. Signs outside this set are OOV and route to the abstain path."
}
```

The map is deliberately **not** identity (`THANK-YOU`→"thank you", `ME`→"i") — exposing it makes
explicit that the translate stage does real lexicon substitution and light reordering, not a
pass-through.

### 2.3 `POST /translate`

The single inference endpoint. It accepts two input modes and runs the full agent FSM.

**Input mode A — `seed` (offline, default).** A reference to a synthetic sequence with embedded gold;
the `SeedPoseEngine`/`SeedRecognizer` read it back. This is the deterministic, dependency-free path
used by tests, the autograder, and the demo.

```json
{
  "mode": "seed",
  "seed": 7,
  "n_signs": 4
}
```

**Input mode B — `frames` (pose keypoints).** A pose-keypoint sequence already in the layout from
`pose/layout.py`: per frame, `2×21` hand landmarks + `25` body landmarks × `3` coords. A client that
ran MediaPipe Holistic itself (browser, edge device) posts the landmarks directly — this is the
privacy-preferred path because **raw video never leaves the device** (§8).

```json
{
  "mode": "frames",
  "frames": [[0.51, 0.42, 0.0, "...per-frame 198-d vector..."]],
  "fps": 25
}
```

**Input mode C — video (full profile only).** When `SIGNLANG_ENABLE_MEDIAPIPE=1`, the endpoint also
accepts a multipart video upload; the server runs MediaPipe Holistic to derive the pose sequence
before entering the FSM (§6). In the default offline image this mode is disabled and returns `503`
with a message pointing the caller at client-side extraction.

**Response.** Glosses, assembled spoken text, per-sign confidence, the segment boundaries, and the
abstain flag — the full agent trace surface:

```json
{
  "abstain": false,
  "needs_review": false,
  "glosses": ["ME", "WANT", "WATER", "THANK-YOU"],
  "text": "i want water thank you",
  "signs": [
    {"gloss": "ME",        "confidence": 0.97, "start_frame": 0,  "end_frame": 14},
    {"gloss": "WANT",      "confidence": 0.91, "start_frame": 19, "end_frame": 33},
    {"gloss": "WATER",     "confidence": 0.88, "start_frame": 38, "end_frame": 52},
    {"gloss": "THANK-YOU", "confidence": 0.95, "start_frame": 57, "end_frame": 73}
  ],
  "low_conf_ratio": 0.0,
  "verification": {"round_trip_gloss": ["ME","WANT","WATER","THANK-YOU"], "chrf_keep": true},
  "n_segments": 4,
  "profile": "seed"
}
```

When the low-confidence/OOV segment ratio exceeds `oov_abstain_ratio` (default `0.5`), the FSM
**abstains** rather than emitting a confident sentence:

```json
{
  "abstain": true,
  "needs_review": true,
  "glosses": ["WATER", "<low-conf>", "<low-conf>"],
  "text": "uncertain",
  "signs": [
    {"gloss": "WATER",     "confidence": 0.84, "start_frame": 0,  "end_frame": 13},
    {"gloss": "<low-conf>","confidence": 0.07, "start_frame": 18, "end_frame": 31},
    {"gloss": "<low-conf>","confidence": 0.05, "start_frame": 36, "end_frame": 50}
  ],
  "low_conf_ratio": 0.67,
  "reason": "low_conf_ratio 0.67 > oov_abstain_ratio 0.50",
  "profile": "seed"
}
```

This is the value-add of the deployment: a returned `"uncertain" + needs_review` is the correct,
honest answer for noisy or out-of-vocabulary signing, and a human is kept in the loop instead of
being handed a fluent hallucination.

**Status codes:** `200` success (including a deliberate abstain — abstention is a valid answer, not an
error); `400` malformed payload (wrong `mode`, frames not in the `pose/layout.py` shape); `413` frame
count above `max_frames`; `422` frame count below `min_frames` (the D1 ingest gate fails); `401`/`403`
auth (§8); `503` video posted to an offline image without MediaPipe.

---

## 3. Gradio demo

`src/signlang/api/gradio_app.py` provides the human-facing UI and calls the same FSM handler, so the
demo and the API can never diverge.

- A **dropdown of pre-canned signed sentences** (the seed sequences) — the user picks one rather than
  having to sign on camera, which keeps the public demo entirely synthetic and **free of any real
  biometric video**. Selecting a sentence shows: the recognized **gloss sequence**, the assembled
  **spoken text**, a **per-sign confidence** bar, and the **abstain banner** when the FSM declines.
- An **"upload pose JSON"** box for `frames` mode, so a developer can paste landmarks extracted
  client-side without sending video to the server.
- A **video upload tab** that is visible only when the full profile is active
  (`SIGNLANG_ENABLE_MEDIAPIPE=1`); it carries an explicit consent + no-retention notice (§8).

Launch:

```bash
python -m signlang.api.gradio_app   # serves on http://localhost:7860
```

---

## 4. Docker

The image must carry the OpenCV/MediaPipe system libraries even when the default runtime profile is
CPU-Seed, because the optional video path links against them. `Dockerfile` (slim Python base):

```dockerfile
FROM python:3.11-slim

# System deps for the MediaPipe video→pose path:
#   ffmpeg   — decode uploaded sign-language video into frames
#   libgl1 / libglib2.0-0 — OpenCV runtime that MediaPipe Holistic links against (libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Default build = offline profile: numpy + fastapi + gradio, NO torch/mediapipe.
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
ENV PYTHONPATH=/app/src \
    SIGNLANG_PROFILE=seed \
    SIGNLANG_ENABLE_MEDIAPIPE=0

EXPOSE 8000
CMD ["uvicorn", "signlang.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- **`ffmpeg`** decodes uploaded video to frames before MediaPipe runs.
- **`libgl1`** (libGL) + **`libglib2.0-0`** are required by the OpenCV runtime that MediaPipe Holistic
  depends on — a slim image omits them and MediaPipe import fails with `libGL.so.1: cannot open shared
  object file`. They are installed unconditionally so the same image can switch to the full profile.
- The **full profile** uses `requirements-full.txt` (adds `torch`, `transformers`, `mediapipe`) and a
  CUDA base image when GPU serving is wanted; build with
  `--build-arg PROFILE=full` / a `Dockerfile.full` variant.

Run:

```bash
docker build -t signlang:seed .
docker run --rm -p 8000:8000 signlang:seed
# full/GPU:
docker run --rm --gpus all -p 8000:8000 \
  -e SIGNLANG_PROFILE=full -e SIGNLANG_ENABLE_MEDIAPIPE=1 signlang:full
```

---

## 5. Hugging Face Space

The public demo deploys as a **Gradio Space running the offline Seed profile on the free CPU tier**:

- No GPU, no torch, no mediapipe → fast cold start and zero per-inference cost.
- The Space only ever serves the **pre-canned synthetic sentences and `frames`-mode pose JSON**, so it
  **stores and transmits no real sign-language video** — the biometric-privacy posture (§8) holds by
  construction.
- A Space `README.md` carries the license flags from the design brief: the demo itself is permissive
  (synthetic data + `google-t5/t5-small` Apache-2.0 + numpy centroid), but the README explicitly notes
  that the **real continuous-SLT corpora are restrictively licensed** — `Exploration-Lab/iSign`
  (CC-BY-NC-SA, gated), `aipieces/How2Sign` & `PSewmuthu/How2Sign_Holistic` (How2Sign CC-BY-NC
  upstream), `Voxel51/WLASL` (license: other) — so none of them are bundled or served.
- If a GPU Space is ever wanted for the transformer + t5 path, it must gate the video tab behind
  consent and retain nothing; the default Space avoids that surface entirely.

---

## 6. The MediaPipe video → pose step (full profile)

When video input is enabled, the server converts video to the pose-keypoint layout **before** the FSM
sees it:

1. **`ffmpeg`** decodes the uploaded clip to frames at the target FPS.
2. **MediaPipe Holistic** (Apache-2.0, Google; frozen/algorithmic — **not trained**) extracts
   hand + body (+ face) landmarks per frame.
3. Landmarks are packed into the `pose/layout.py` vector: `2×21` hand + `25` body × `3` coords — the
   **same shape** `PSewmuthu/How2Sign_Holistic` produces — and handed to D1 ingest.

This stage is **CPU-bound and stateless**: MediaPipe Holistic runs comfortably on CPU and benefits
little from a GPU, so even in the full profile the front-end stays on CPU while only the trainable core
moves to GPU (§7 split). In the **offline profile this step does not run at all** — the
`SeedPoseEngine` synthesizes/reads the pose sequence directly, which is why the default image needs
neither ffmpeg-at-runtime nor mediapipe.

The recommended production pattern is **client-side / on-device MediaPipe**: the device extracts
landmarks and posts `frames`, so raw biometric video never reaches the server.

---

## 7. GPU vs CPU serving

The cascade is deliberately split so the expensive part is small and isolable:

| Stage | Offline profile | Full profile | Notes |
|---|---|---|---|
| Pose front-end | SeedPoseEngine — **CPU** | MediaPipe Holistic — **CPU** | algorithmic; GPU gives little benefit |
| Segmentation (D2) | velocity threshold — **CPU** | velocity threshold — **CPU** | pure numpy on landmark velocities |
| Recognizer (D3) | numpy nearest-centroid — **CPU** | transformer encoder — **GPU** | the ~5M pose model (cf. `manohonsy/how2sign-pose-cslr`, 4.8M) |
| Translator (D4) | lexicon map — **CPU** | `google-t5/t5-small` (60.5M) — **GPU** | t5 decode is the heaviest single op |
| Verify + finalize | chrF / ratios — **CPU** | chrF / ratios — **CPU** | cheap |

So: **pose extraction, segmentation, the numpy centroid, the lexicon, and the agent logic all run on
CPU**; only the **transformer recognizer and the t5-small translator** want a GPU, and only in the
full profile. A single small GPU (a T4 is ample — t5-small is 60.5M and the recognizer ~5M) serves the
full profile; the default Seed profile needs **no GPU at all**. This keeps the offline path
genuinely CPU-deployable on the free Space and in CI, and keeps the GPU footprint minimal when the
trained core is used.

---

## 8. Latency

Indicative single-request latency (short sentence, 2–6 signs, ~60–120 frames):

| Profile | Stage | Latency |
|---|---|---|
| **Seed (CPU)** | full request (no video, no torch) | **~3–10 ms** |
| Full (CPU front-end) | ffmpeg decode + MediaPipe Holistic | ~30–60 ms/sec of video |
| Full (GPU core) | transformer recognizer (per segment) | ~3–8 ms/segment |
| Full (GPU core) | t5-small gloss→text decode | ~20–60 ms/sentence |

The offline path is effectively instant (numpy centroid + dict lookup). In the full profile the
**MediaPipe video→pose step dominates** and scales with clip length, which is the operational reason
to push pose extraction to the client/edge: the server then only pays the small GPU core cost. Latency
budgets should be set per profile, and clients should expect a deliberate **abstain to return just as
fast** as a confident answer — abstention is a cheap branch, not a timeout.

---

## 9. Configuration via environment variables

All knobs are env-driven (12-factor; the shared config template). No secrets in code.

| Variable | Default | Meaning |
|---|---|---|
| `SIGNLANG_PROFILE` | `seed` | `seed` (offline CPU) or `full` (trained core) |
| `SIGNLANG_ENABLE_MEDIAPIPE` | `0` | enable the video→pose endpoint mode (needs mediapipe + ffmpeg + libGL) |
| `SIGNLANG_DEVICE` | `cpu` | `cpu` or `cuda` for the transformer recognizer + t5 |
| `SIGNLANG_TRANSLATOR_ID` | `google-t5/t5-small` | translator checkpoint (Apache-2.0); alt `facebook/m2m100_418M` (MIT), `google/byt5-small` (Apache-2.0) |
| `SIGNLANG_MIN_FRAMES` | `8` | D1 ingest gate: reject sequences shorter than this (`422`) |
| `SIGNLANG_MAX_FRAMES` | `4000` | upper frame bound (`413`) |
| `SIGNLANG_RECOG_MIN_CONF` | `0.15` | D3 per-segment confidence floor; below → low-confidence |
| `SIGNLANG_OOV_ABSTAIN_RATIO` | `0.5` | D5 abstain when low-conf/OOV segment ratio exceeds this |
| `SIGNLANG_REST_VELOCITY` | `0.02` | D2 motion threshold: frames below this split signs |
| `SIGNLANG_ENABLE_LLM_BRAIN` | `0` | optional advisory `anthropic` brain; **off by default**, never changes output |
| `ANTHROPIC_API_KEY` | unset | only read when the LLM brain is explicitly enabled |
| `SIGNLANG_REQUIRE_AUTH` | `1` | require an API key on `/translate` when video/`frames` carry biometric pose |
| `SIGNLANG_API_KEY` | unset | the bearer/API key checked when auth is required |
| `SIGNLANG_RETAIN_INPUTS` | `0` | **off** — never persist uploaded video/pose; in-memory only |

The recognizer confidence floor, abstain ratio, rest-velocity threshold, and frame gates are the same
constants the agent FSM uses (`recog_min_conf=0.15`, `oov_abstain_ratio=0.5`), exposed so operators can
tune the abstention aggressiveness per deployment context (a medical/legal setting should set a higher
`OOV_ABSTAIN_RATIO` and `RECOG_MIN_CONF`).

---

## 10. Request / response example (end to end)

Confident case, offline Seed profile:

```bash
curl -s -X POST http://localhost:8000/translate \
  -H "Authorization: Bearer $SIGNLANG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"mode": "seed", "seed": 7, "n_signs": 4}'
```

```json
{
  "abstain": false,
  "needs_review": false,
  "glosses": ["ME", "WANT", "WATER", "THANK-YOU"],
  "text": "i want water thank you",
  "signs": [
    {"gloss": "ME",        "confidence": 0.97, "start_frame": 0,  "end_frame": 14},
    {"gloss": "WANT",      "confidence": 0.91, "start_frame": 19, "end_frame": 33},
    {"gloss": "WATER",     "confidence": 0.88, "start_frame": 38, "end_frame": 52},
    {"gloss": "THANK-YOU", "confidence": 0.95, "start_frame": 57, "end_frame": 73}
  ],
  "low_conf_ratio": 0.0,
  "n_segments": 4,
  "profile": "seed"
}
```

Client-extracted pose (`frames` mode), full profile with GPU core:

```bash
curl -s -X POST http://localhost:8000/translate \
  -H "Authorization: Bearer $SIGNLANG_API_KEY" \
  -H "Content-Type: application/json" \
  -d @pose_sequence.json   # {"mode":"frames","frames":[[...198-d...]], "fps":25}
```

Note the `text` (`"i want water thank you"`) differs from the raw gloss tokens
(`ME WANT WATER THANK-YOU`) — the lexicon map and casing are doing real work, which is exactly the
margin the BLEU/identity baseline in `translation_evaluation.md` measures.

> **Metric honesty (carried into the API docs):** automatic SLT metrics (BLEU/chrF/ROUGE/BLEURT)
> are unreliable — length-sensitive and blind to hallucination / semantic equivalence (Yazdani et
> al. 2025, hf.co/papers/2510.25434). A served `text` should never be presented to an end user as a
> verified translation; the per-sign confidence and `abstain` flag are the trustworthy signals.

---

## 11. Biometric-privacy serving notes

Sign-language video is **biometric and identifying** — it captures faces and hands and is
**Deaf-community data**. The deployment treats this as a first-class constraint, not an afterthought:

- **No retention by default.** `SIGNLANG_RETAIN_INPUTS=0`: uploaded video and pose are processed
  in-memory and discarded after the response. No request body — video, frames, or derived landmarks —
  is written to disk or logs. Logs record only metadata (timing, profile, abstain flag, gloss/text
  lengths), never the pose payload.
- **Prefer on-device / edge extraction.** The privacy-preferred path is **client-side MediaPipe**: the
  device extracts landmarks and posts `frames`, so raw video never reaches the server. The video
  endpoint is **off by default** (`SIGNLANG_ENABLE_MEDIAPIPE=0`).
- **Authentication.** `/translate` requires an API key when it carries biometric pose
  (`SIGNLANG_REQUIRE_AUTH=1`, bearer token). `/healthz` and `/vocabulary` are unauthenticated (no
  user data). Serve over TLS; the public Space serves only synthetic pre-canned sentences and so
  carries no real biometric data.
- **Consent.** Any UI that accepts real video (the full-profile Gradio video tab) shows an explicit
  consent + no-retention notice before capture.
- **LLM brain off.** The optional `anthropic` advisory brain is **disabled by default** and never sees
  raw pose; even when enabled it is advisory (a "please repeat" hint) and **never changes the output**,
  so no biometric data is sent to a third-party API in the default deployment.
- **Never present low-confidence output as authoritative.** Representation bias is acute — a model
  trained on one sign language / signer set fails on others. The server therefore always returns
  **per-sign confidence and the abstain flag**, and in medical/legal contexts operators should raise
  `RECOG_MIN_CONF` / `OOV_ABSTAIN_RATIO` and keep a **human interpreter in the loop**. The system is an
  **assistive aid, not a replacement** for human interpreters.
```