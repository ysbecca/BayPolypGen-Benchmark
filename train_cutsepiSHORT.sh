#!/bin/bash
#SBATCH --account=bdlds05
#SBATCH --time=48:0:0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

module load cuda
source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis

export WANDB_MODE=online
export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

CUTS=(0 \
	59 \
	29 \
	15 \
	116 \
)



CYCLE_LENGTHS=(300 400)

task_id=0

for c in "${CUTS[@]}"
do
	for cl in "${CYCLE_LENGTHS[@]}"
	do
		if [ $task_id = $SLURM_ARRAY_TASK_ID ]
		then
			echo $c
			echo $k
			python main_polypGen.py \
				--cycle_length $cl \
				--alpha 0.9 \
				--cycles 2 \
				--kappa 3 \
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
