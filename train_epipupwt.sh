#!/bin/bash
#SBATCH --account=bdlds05
#SBATCH --time=24:0:0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-2



 
module load cuda

source /nobackup/projects/bdlds05/rsstone/miniconda/etc/profile.d/conda.sh
conda activate pyvis

export WANDB_MODE=online
export WANDB_DIR="/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/"


KAPPAS=(2 5 7)
task_id=0

BASELINE_MODEL_DESC="lively-moon-53"

for k in "${KAPPAS[@]}"
do
	if [ $task_id = $SLURM_ARRAY_TASK_ID ]
	then
		# TODO fill in params of best baseline bayesian model
		python main_polypGen.py \
			--cycle_length 550 \
			--cycles 1 \
			--epiupwt True \
			--alpha 0.9 \
			--kappa $k \
			--models_per_cycle 10 \
			--model_desc $BASELINE_MODEL_DESC \
			--model "deeplabv3plus_resnet50" \
			--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
			--lr 0.1
		exit 0
	fi
	let task_id=$task_id+1
done


# python main_polypGen.py \
# 	--cycle_length 2 \
# 	--cycles 1 \
# 	--epiupwt True \
# 	--kappa 5 \
# 	--models_per_cycle 2 \
# 	--model_desc "test" \
# 	--model "deeplabv3plus_resnet50" \
# 	--root "/users/rsstone/projects_sym/rsstone/BayPolypGen-Benchmark/" \
# 	--lr 0.1
