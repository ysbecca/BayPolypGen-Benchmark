#!/bin/bash

# test bayesian train no de-biasing



source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis
export WANDB_MODE=offline
export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

python main_polypGen.py \
	--cycle_length 3 \
	--cycles 1 \
	--batch_size 4 \
	--models_per_cycle 2 \
	--model "deeplabv3plus_resnet50" \
	--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
	--lr 0.1
wandb sync
