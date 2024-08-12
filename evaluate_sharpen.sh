#!/bin/bash
#SBATCH --account=bdlds05
#SBATCH --time=2:00:0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-26

module load cuda

source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis

wandb online
export WANDB_MODE=online
export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"
ROOT="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"


SHARPEN_MODELS=("driven-sun-53")

EPOCHS=(0 1 2)

DATASETS=("C6_pred" "EndoCV_DATA3" "EndoCV_DATA4")

LRS=(0.1 0.05 0.01)

task_id=0


for model in "${SHARPEN_MODELS[@]}"
do
	for dst in "${DATASETS[@]}"
	do
		for e in "${EPOCHS[@]}"
		do
			if [ $task_id = $SLURM_ARRAY_TASK_ID ]
			then
				for lr in "${LRS[@]}"
				do
					python polypGen_inference-seg.py \
						--moment_count 5 \
						--test_set $dst \
						--is_sharpen True \
						--epoch $e \
						--lr $lr \
						--model_desc $model \
						--root $ROOT

					python metrics/compute_seg.py \
						--model_desc $model \
						--test_set $dst \
						--is_sharpen True \
						--epoch $e \
						--lr $lr \
						--root $ROOT
				done
				echo "inference and evaluation done - sharpen."
				echo $model
				echo $dst
				exit 0
			fi
			let task_id=$task_id+1
		done
	done
done



# python polypGen_inference-seg.py --moment_count 7 --model_desc "driven-sun-53" --root /users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/
# python metrics/compute_seg.py --model_desc "driven-sun-53" --root /users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/
