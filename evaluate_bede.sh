#!/bin/bash
#SBATCH --account=bdlds05
#SBATCH --time=3:00:0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-17

module load cuda

source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis

export WANDB_MODE=online

# TODO 1: fill in the MODELS list with all the string model descs you want to evaluate
MODELS=("unique-durian-1" \
	"dainty-darkness-2" \
	"polar-planet-3" \
	"sparkling-microwave-4" \
)
SHARPEN_MODELS=("prime-water-15")

EPOCHS=(0 1)

DATASETS=("C6_pred" "EndoCV_DATA3" "EndoCV_DATA4")

# TODO 2: fill in these paths
export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"
ROOT="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"

# TODO 3: if there are N models, set task array count to N-1 in line 7 for SLURM; 
# or replace with Sun Grid Engine syntax
task_id=0

for model in "${MODELS[@]}"
do
	for dst in "${DATASETS[@]}"
	do
		if [ $task_id = $SLURM_ARRAY_TASK_ID ]
		then
			python polypGen_inference-seg.py \
				--moment_count 6 \
				--test_set $dst \
				--model_desc $model \
				--root $ROOT

			# this logs test metrics to wandb
			python metrics/compute_seg.py \
				--model_desc $model \
				--test_set $dst \
				--root $ROOT

			echo "inference and evaluation done."
			echo $model
			echo $dst
			exit 0
		fi
		let task_id=$task_id+1
	done
done

echo "now evaluating sharpen models"

for model in "${SHARPEN_MODELS[@]}"
do
	for dst in "${DATASETS[@]}"
	do
		for e in "${EPOCHS[@]}"
		do
			if [ $task_id = $SLURM_ARRAY_TASK_ID ]
			then
				python polypGen_inference-seg.py \
					--moment_count 6 \
					--test_set $dst \
					--is_sharpen True \
					--epoch $e \
					--model_desc $model \
					--root $ROOT

				python metrics/compute_seg.py \
					--model_desc $model \
					--test_set $dst \
					--is_sharpen True \
					--epoch $e \
					--root $ROOT

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
