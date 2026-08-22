# Edge detection and tracking.  `make demo` is the one command; `make check` is
# exactly what CI runs.
#
# Four classes of target and the distinction is the honest part of this file:
#
#   FAST     seconds to a couple of minutes, no dataset, no GPU, no weights.
#            Everything CI runs on every push.
#              make check  make test  make lint  make demo  make docs-check
#
#   DATA     regenerates the dataset from its seeds.  About seventy seconds for
#            the published preset, entirely deterministic.
#              make data  make data-tiny
#
#   HEAVY    needs a CUDA GPU and the training toolchain.  `make experiments`
#            regenerates every number quoted in the README; roughly twenty five
#            minutes end to end on an RTX 4060.
#              make train  make reid  make export  make quantise
#              make evaluate  make track  make bench  make experiments
#
#   JETSON   needs a Jetson with TensorRT.  Never run: no board was available.
#            No number in this repository came from these targets and
#            docs/jetson.md says so in more detail.
#              make engine  make bench-jetson

SHELL := /bin/sh
PYTHON ?= python
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
ifeq ($(wildcard $(VENV_PY)),)
VENV_PY := $(VENV)/Scripts/python.exe
endif
ifeq ($(wildcard $(VENV_PY)),)
VENV_PY := $(PYTHON)
endif

DATA ?= data/processed
MODELS ?= models
RESULTS ?= results
PUBLISHED := $(RESULTS)/published
NAME ?= yolov8n_edgetrack
FP32 := $(MODELS)/$(NAME)_fp32.onnx
FP16 := $(MODELS)/$(NAME)_fp16.onnx
INT8 := $(MODELS)/$(NAME)_int8.onnx
INT8_ALLCONV := $(MODELS)/$(NAME)_int8_allconv.onnx
INT8_ALLOPS := $(MODELS)/$(NAME)_int8_allops.onnx
CHECKPOINT := $(MODELS)/$(NAME).pt
REID := $(MODELS)/reid_edgetrack.onnx

PRESET ?= full
SOURCE ?= synth:7
TRACKER ?= bytetrack
ITERATIONS ?= 300

export PYTHONPATH := src

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  python: $(VENV_PY)"

# ---------------------------------------------------------------------------
# Fast
# ---------------------------------------------------------------------------

.PHONY: check
check: lint test docs-check  ## Everything CI runs on every push

.PHONY: lint
lint:  ## flake8 over src, tests and scripts
	$(VENV_PY) -m flake8 src/ tests/ scripts/

.PHONY: test
test:  ## The whole test suite
	$(VENV_PY) -m pytest tests/ -q --durations=15

.PHONY: test-fast
test-fast:  ## Tests that need neither weights nor a dataset
	$(VENV_PY) -m pytest tests/ -q -m "not needs_model and not slow"

.PHONY: demo
demo: | $(FP32)  ## Detect, track, score and render one sequence
	$(VENV_PY) scripts/run_demo.py --source $(SOURCE) --model $(FP32) \
	  --tracker $(TRACKER) --video $(RESULTS)/demo.mp4 --gif docs/figures/demo.gif \
	  --out $(RESULTS)/demo.json

.PHONY: docs-check
docs-check:  ## Every number in the docs traces to results/published/
	$(VENV_PY) scripts/check_docs.py

.PHONY: docs-fix
docs-fix:  ## Rewrite the generated tables embedded in the docs
	$(VENV_PY) scripts/make_results_table.py --quiet
	$(VENV_PY) scripts/analyse_idswitches.py --tracking $(PUBLISHED)/tracking.json \
	  --json-out $(PUBLISHED)/idswitches.json --quiet
	$(VENV_PY) scripts/check_docs.py --fix-blocks

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

.PHONY: data
data:  ## Generate the dataset ($(PRESET) preset)
	$(VENV_PY) scripts/make_dataset.py --preset $(PRESET) --root $(DATA)

.PHONY: data-tiny
data-tiny:  ## The smallest dataset that exercises the pipeline
	$(VENV_PY) scripts/make_dataset.py --preset tiny --root $(DATA)

$(DATA)/data.yaml:
	$(MAKE) data

# ---------------------------------------------------------------------------
# Heavy
# ---------------------------------------------------------------------------

.PHONY: model
model:  ## Fetch the published weights from the GitHub Release
	$(VENV_PY) scripts/fetch_model.py

$(FP32):
	@echo "no $(FP32); run 'make model' to download it or 'make train export' to build it" >&2
	@exit 1

.PHONY: train
train: $(DATA)/data.yaml  ## Fine-tune YOLOv8n (needs a CUDA GPU)
	$(VENV_PY) scripts/train.py --config configs/train.yaml --data $(DATA)/data.yaml \
	  --out $(MODELS) --name $(NAME)

.PHONY: reid
reid:  ## Train and export the appearance embedding
	$(VENV_PY) scripts/train_reid.py --out $(MODELS)/reid_edgetrack.pt
	$(VENV_PY) scripts/export_reid.py --weights $(MODELS)/reid_edgetrack.pt --out $(REID)

.PHONY: export
export:  ## Export FP32 and FP16 ONNX from the checkpoint
	$(VENV_PY) scripts/export_onnx.py --weights $(CHECKPOINT) --precision fp32
	$(VENV_PY) scripts/export_onnx.py --weights $(CHECKPOINT) --precision fp16

.PHONY: quantise
quantise: $(FP32)  ## Build the three INT8 variants
	$(VENV_PY) scripts/quantize_onnx.py --model $(FP32) --data-root $(DATA)

.PHONY: evaluate
evaluate:  ## Detection accuracy for every precision variant
	$(VENV_PY) scripts/evaluate.py --split test --data-root $(DATA) \
	  --providers CPUExecutionProvider --out $(RESULTS)/detection.json \
	  --models $(FP32) $(FP16) $(INT8) $(INT8_ALLCONV) $(INT8_ALLOPS)

.PHONY: tune
tune:  ## Choose tracker parameters on the val split
	$(VENV_PY) scripts/tune_tracker.py --tracker bytetrack --model $(FP32) --data-root $(DATA)
	$(VENV_PY) scripts/tune_tracker.py --tracker deepsort --model $(FP32) --data-root $(DATA) \
	  --appearance reid --reid-model $(REID)

.PHONY: track
track:  ## MOTA, IDF1 and ID switches for both trackers
	$(VENV_PY) scripts/eval_tracking.py --model $(FP32) --data-root $(DATA) \
	  --trackers bytetrack deepsort --appearance reid --reid-model $(REID) \
	  --out $(RESULTS)/tracking.json
	$(VENV_PY) scripts/eval_tracking.py --model $(FP32) --data-root $(DATA) \
	  --trackers bytetrack deepsort --appearance histogram \
	  --out $(RESULTS)/tracking_histogram.json
	$(VENV_PY) scripts/eval_tracking.py --detections gt --data-root $(DATA) \
	  --trackers bytetrack deepsort --appearance reid --reid-model $(REID) \
	  --out $(RESULTS)/tracking_gt.json
	$(VENV_PY) scripts/analyse_idswitches.py --tracking $(RESULTS)/tracking.json \
	  --json-out $(RESULTS)/idswitches.json

.PHONY: probe
probe:  ## Is the GPU number measuring the model or the kernel launches?
	$(VENV_PY) scripts/probe_launch_bound.py --weights $(CHECKPOINT) \
	  --out $(RESULTS)/launch_bound.json

.PHONY: bench
bench:  ## Latency and throughput on this host
	$(VENV_PY) scripts/benchmark_latency.py --variants configs/benchmark.yaml \
	  --data-root $(DATA) --iterations $(ITERATIONS) --skip-missing \
	  --out $(RESULTS)/latency.json

.PHONY: experiments
experiments: evaluate track bench probe  ## Regenerate every published number
	@echo
	@echo "Results are in $(RESULTS)/.  To publish them:"
	@echo "  cp $(RESULTS)/*.json $(PUBLISHED)/ && make docs-fix"

.PHONY: publish
publish:  ## Copy this run's results into results/published and refresh the tables
	cp $(RESULTS)/detection.json $(RESULTS)/tracking.json \
	   $(RESULTS)/tracking_histogram.json $(RESULTS)/tracking_gt.json \
	   $(RESULTS)/latency.json $(RESULTS)/demo.json $(RESULTS)/launch_bound.json \
	   $(RESULTS)/reid_train.json $(RESULTS)/idswitches.json \
	   $(RESULTS)/tuning_bytetrack.json $(RESULTS)/tuning_deepsort.json $(PUBLISHED)/
	cp $(DATA)/dataset.json $(PUBLISHED)/dataset.json
	cp $(MODELS)/quantisation.json $(PUBLISHED)/quantisation.json
	cp $(MODELS)/$(NAME)_train.json $(PUBLISHED)/train.json
	$(MAKE) docs-fix
	$(MAKE) docs-check

# ---------------------------------------------------------------------------
# Jetson.  Never executed; see docs/jetson.md.
# ---------------------------------------------------------------------------

.PHONY: engine
engine:  ## Build TensorRT engines (Jetson only, never run)
	$(VENV_PY) scripts/export_tensorrt.py --onnx $(FP32) --precision fp32
	$(VENV_PY) scripts/export_tensorrt.py --onnx $(FP32) --precision fp16
	$(VENV_PY) scripts/export_tensorrt.py --onnx $(FP32) --precision int8

.PHONY: bench-jetson
bench-jetson:  ## Benchmark on a Jetson (never run)
	sh scripts/bench_jetson.sh

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

.PHONY: docker-runtime
docker-runtime:  ## Build the runtime image and run the demo inside it
	docker build -f docker/Dockerfile.runtime -t edgetrack:runtime .
	docker run --rm -v "$(PWD)/results:/app/results" edgetrack:runtime \
	  python scripts/run_demo.py --source synth:7 --frames 40 \
	    --model models/yolov8n_edgetrack_fp32.onnx --out /app/results/docker_demo.json

.PHONY: docker-train
docker-train:  ## Build the training image
	docker build -f docker/Dockerfile.train -t edgetrack:train .

.PHONY: up
up:  ## docker compose up: the demo, end to end
	docker compose -f docker/docker-compose.yml up --build

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: clean
clean:  ## Remove run outputs, keeping results/published and the dataset
	rm -rf $(RESULTS)/*.json $(RESULTS)/*.mp4 $(RESULTS)/mot .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean  ## Also remove the dataset and every model artefact
	rm -rf $(DATA) $(MODELS)/*.onnx $(MODELS)/*.pt $(MODELS)/*.engine $(MODELS)/runs
