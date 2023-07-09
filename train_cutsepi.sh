#!/bin/bash
#SBATCH --account=bdlds05
#SBATCH --time=48:0:0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

module load cuda
source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis

export WANDB_MODE=online
export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

CUTS=(116 \
	58 \
	29 \
	15 \
)

KAPPAS=(2 5)

task_id=0

for c in "${CUTS[@]}"
do
	for k in "${KAPPAS[@]}"
	do
		if [ $task_id = $SLURM_ARRAY_TASK_ID ]
		then
			echo $c
			echo $k
			echo $e
			python main_polypGen.py \
				--cycle_length 550 \
				--alpha 0.9 \
				--cycles 1 \
				--kappa $k \
				--epiupwt True \
				--extra_C6 $c \
				--models_per_cycle 10 \
				--model "deeplabv3plus_resnet50" \
				--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
				--lr 0.1
			exit 0
		fi
		let task_id=$task_id+1
	done
done
