# Portable targets. Same names as make.ps1. Data root = $ANC_DATA_ROOT (default ./data).
PY ?= python
CFG ?= configs/train.yaml

.PHONY: setup download download-all rir registry smoke test selftest evals dump datasets baseline plots docker help

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/  -/'

setup:        ## venv-less install (run inside your venv / container)
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

download:     ## LibriSpeech + ESC-50 (background-safe, idempotent)
	bash scripts/download_sources.sh

download-all: ## + Zenodo gunshots, MAD (kaggle cli), RIRS_NOISES
	bash scripts/download_sources.sh --all

rir:          ## simulate the RIR bank (300 rooms; N_PER_PRESET=400 for 2000)
	$(PY) -m ancdata.cli rir --n-per-preset $(or $(N_PER_PRESET),60)

registry:     ## index sources, hash splits, voice screen -> manifest.parquet
	$(PY) -m ancdata.cli registry

smoke:        ## no-download end-to-end check (< 60 s)
	$(PY) -m ancdata.cli selftest --smoke

test:         ## unit tests (includes smoke)
	$(PY) -m pytest -q

selftest:     ## the 8 checks on real data
	$(PY) -m ancdata.cli selftest --config $(CFG)

evals:        ## freeze the three synthetic eval sets
	$(PY) -m ancdata.cli materialize --config configs/eval_standard.yaml --split test --n 2000
	$(PY) -m ancdata.cli materialize --config configs/eval_generalization.yaml --split test --n 1000
	-$(PY) -m ancdata.cli materialize --config configs/eval_lombard.yaml --split test --n 500

TRAIN_N ?= 10000
VAL_N ?= 500
dump:         ## materialised training dump (FLAC, ~0.4 GB per 1000 pairs): TRAIN_N=10000 VAL_N=500
	$(PY) -m ancdata.cli materialize --config $(CFG) --split train --n $(TRAIN_N) --fmt flac --out $(or $(ANC_DATA_ROOT),data)/train_dump
	$(PY) -m ancdata.cli materialize --config $(CFG) --split val   --n $(VAL_N)   --fmt flac --out $(or $(ANC_DATA_ROOT),data)/val_dump

datasets:     ## everything for the model team: registry, selftest, evals, dump, baseline, plots
	$(MAKE) registry selftest evals dump baseline plots

baseline:     ## unprocessed-input metrics table for every frozen set
	for d in $$ANC_DATA_ROOT/eval/* data/eval/*; do [ -f $$d/meta.jsonl ] && $(PY) -m ancdata.cli evaluate $$d; done; true

plots:        ## crest/attack plot (reality-gap needs real recordings: see README)
	$(PY) -m ancdata.cli plots --config $(CFG)

docker:       ## build the image for the workstation
	docker build -t ancdata .
