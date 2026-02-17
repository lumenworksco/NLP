.PHONY: all data train evaluate test clean help

PYTHON ?= python
JUPYTER ?= jupyter nbconvert --to notebook --execute --inplace

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

all: data train evaluate  ## Run full pipeline (data → train → evaluate)

data:  ## Run notebook 01: download and preprocess datasets
	$(JUPYTER) 01_data_exploration.ipynb

train: data  ## Run notebooks 02–03: train all models
	$(JUPYTER) 02_monolingual_finetuning.ipynb
	$(JUPYTER) 03_multilingual_finetuning.ipynb

evaluate: train  ## Run notebook 04: evaluate and compare models
	$(JUPYTER) 04_evaluation_comparison.ipynb

test:  ## Run unit tests
	$(PYTHON) -m pytest tests/ -v

clean:  ## Remove model checkpoints and generated results
	rm -rf models/*
	rm -f results/*.csv
	rm -f results/figures/f1_heatmap_all_models.png
	rm -f results/figures/crosslingual_heatmap.png
	rm -f results/figures/monolingual_vs_multilingual.png
	rm -f results/figures/confusion_matrices_mbert_all.png
