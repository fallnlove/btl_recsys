#!/bin/bash

#SBATCH --job-name=ml_optuna
#SBATCH --error=results/random-optuna-%A-%a.err
#SBATCH --output=results/random-optuna-%A-%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-15

MODEL=ultragcn

echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "Model = $MODEL"

module load Python
conda activate recsys_btl

python3 sbatch/ultragcn/run.py \
    --dataset_idx "$SLURM_ARRAY_TASK_ID" \
    --model "$MODEL"