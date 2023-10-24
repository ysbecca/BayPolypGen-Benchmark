#!/bin/bash
#SBATCH --account=bdlds05
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

# test bayesian train no de-biasing

echo $SLURM_ARRAY_TASK_ID

module load cuda

source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis

export WANDB_MODE=online
export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

CYCLE_LENGTH=(250 550)
ALPHAS=(0.7 0.8 0.9)
task_id=0

for c in "${CYCLE_LENGTH[@]}"
do
	for a in "${ALPHAS[@]}"
	do
		if [ $task_id = $SLURM_ARRAY_TASK_ID ]
		then
			python main_polypGen.py \
				--cycle_length $c \
				--cycles 1 \
				--models_per_cycle 10 \
				--model "deeplabv3plus_resnet50" \
				--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
				--lr 0.1 \
				--alpha $a
			exit 0
		fi
		let task_id=$task_id+1
	done
done
