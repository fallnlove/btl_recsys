#!/bin/bash

#SBATCH --job-name=ml_optuna
#SBATCH --error=results/pure_svd-optuna-%A-%a.err
#SBATCH --output=results/pure_svd-optuna-%A-%a.log
#SBATCH --time=4:10:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=2,4,5,6,7,8,9
#SBATCH --cpus-per-task=4
#SBATCH --constraint="type_e|type_f"


MODEL=pure_svd

echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "Model = $MODEL"

module load Python
conda activate recsys

python3 sbatch/pure_svd/run.py \
    --dataset_idx "$SLURM_ARRAY_TASK_ID" \
    --model "$MODEL" \
