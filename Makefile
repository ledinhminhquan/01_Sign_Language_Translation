.PHONY: help install install-all test lint data synthetic train evaluate translate demo serve report slides autopilot grade docker clean

PY ?= python

help:
	@echo "Sign Language Translation (signlang) — common targets:"
	@echo "  install      core deps + package (no torch/mediapipe)"
	@echo "  install-all  package with [all] extras (ml+pose+api+report)"
	@echo "  test         run the offline pytest suite"
	@echo "  data         prefetch/sanity-check (backbone + real smoke corpus + seed)"
	@echo "  synthetic    render a synthetic pose-sequence collection"
	@echo "  train        train the recognizer (+ t5 translator) (needs a GPU)"
	@echo "  evaluate     gloss WER/acc + BLEU/chrF vs baselines (--fast = offline centroid)"
	@echo "  translate S= translate a synthetic signed sentence by seed"
	@echo "  demo         run the agent on the held-out synthetic split"
	@echo "  serve        run the FastAPI server + Gradio UI"
	@echo "  autopilot    one-button: train->eval->analysis->report+slides+grade+bundle"
	@echo "  grade        rubric completeness self-check"

install:
	$(PY) -m pip install -e .

install-all:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests

data:
	signlang data

synthetic:
	signlang gen-synthetic

train:
	signlang train

evaluate:
	signlang evaluate --fast

translate:
	signlang translate --seed "$(S)" --fast

demo:
	signlang demo-agent --fast

serve:
	signlang serve --ui

report:
	signlang generate-report

slides:
	signlang generate-slides

autopilot:
	signlang autopilot

grade:
	signlang grade

docker:
	docker build -t signlang:latest .

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
