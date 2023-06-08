#!/bin/bash


# test bayesian train no de-biasing



python main_polypGen.py \
	--cycle_length 3 \
	--cycles 1 \
	--models_per_cycle 1 \
	--model "deeplabv3plus_resnet50" \
	--root "/usr/not-backed-up/BayPolypGen-Benchmark/datasets/" \
	--lr 0.1
	# --dev_run True  # doesn't create wandb run or save checkpoints

