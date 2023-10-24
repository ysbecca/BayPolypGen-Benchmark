#!/bin/bash
	# # SBATCH --account=bdlds05
	# # SBATCH --time=48:0:0
	# # SBATCH --partition=gpu
	# # SBATCH --nodes=1
	# # SBATCH --gres=gpu:1

	# # test bayesian train no de-biasing


	# module load cuda

source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis

	# export WANDB_MODE=online
	# export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

# python main_polypGen.py \
# 	--cycle_length 350 \
# 	--cycles 1 \
# 	--models_per_cycle 10 \
# 	--model "deeplabv3plus_resnet50" \
# 	--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
# 	--lr 0.1 


# test run to save train epis only

python main_polypGen.py \
	--cycle_length 0 \
	--cycles 0 \
	--model_desc "fresh-paper-72" \
	--moment_count 5 \
	--extra_C6 29 \
	--model "deeplabv3plus_resnet50" \
	--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
	--lr 0.1
