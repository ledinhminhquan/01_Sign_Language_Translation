# Sign Language Translation — API + Gradio UI image.
# ffmpeg + libGL are needed for video decoding + MediaPipe; the seq2seq core runs on CPU here
# (mount a GPU + install torch CUDA for accelerated recognition/translation).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SIGNLANG_ARTIFACTS_DIR=/data/artifacts \
    HF_HOME=/data/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e ".[api,report]"

RUN mkdir -p /data/artifacts /data/hf
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)" || exit 1

CMD ["signlang", "serve", "--host", "0.0.0.0", "--port", "8000", "--ui"]
