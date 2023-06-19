#!/bin/bash


# test bayesian train no de-biasing


export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

python main_polypGen.py \
	--cycle_length 3 \
	--cycles 1 \
	--models_per_cycle 2 \
	--batch_size 3 \
	--model "deeplabv3plus_resnet50" \
	--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
	--lr 0.1 \
	--dev_run True  # doesn't create wandb run or save checkpoints

