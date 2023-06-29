#!/bin/bash
# #SBATCH --account=bdlds05
# #SBATCH --time=48:0:0
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --gres=gpu:1
# #SBATCH --array=0-




# module load cuda

# source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
# conda activate pyvis

# export WANDB_MODE=online
# export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

# task_id=0



BASELINE_MODEL_DESC="driven-sun-53"


# for lr in "${LRS[@]}"
# do
# 	if [ $task_id = $SLURM_ARRAY_TASK_ID ]
# 	then
python sharpen.py \
	--model_desc $BASELINE_MODEL_DESC \
	--max_epochs 2 \
	--moment_count 2 \
	--batch_size 2 \
	--lr 0.1 \
	--root "/usr/not-backed-up/BayPolypGen-Benchmark/"
	# --root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
# 		exit 0
# 	fi
# 	let task_id=$task_id+1
# done