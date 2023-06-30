#!/bin/bash
#SBATCH --account=bdlds05
#SBATCH --time=48:0:0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-1




module load cuda

source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis

export WANDB_MODE=online
export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

task_id=0

BASELINE_MODELS=("driven-sun-53") #"legendary-moon-51")
LRS=(0.1 0.05 0.01)

for m in "${BASELINE_MODELS[@]}"
do
	for lr in "${LRS[@]}"
	do
		if [ $task_id = $SLURM_ARRAY_TASK_ID ]
		then
			python sharpen.py \
				--model_desc $m \
				--max_epochs 20 \
				--moment_count 3 \
				--batch_size 12 \
				--lr $lr \
				--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"
				# --root "/usr/not-backed-up/BayPolypGen-Benchmark/"

			exit 0
		fi
		let task_id=$task_id+1
	done
done