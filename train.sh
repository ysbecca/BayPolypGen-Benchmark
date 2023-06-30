#!/bin/bash
#SBATCH --account=bdlds05
#SBATCH --time=48:0:0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# test bayesian train no de-biasing


module load cuda

source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis

export WANDB_MODE=online
export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

python main_polypGen.py \
	--cycle_length 350 \
	--cycles 1 \
	--models_per_cycle 10 \
	--model "deeplabv3plus_resnet50" \
	--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
	--lr 0.1 
