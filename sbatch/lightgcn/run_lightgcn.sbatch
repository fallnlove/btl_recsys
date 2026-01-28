#!/bin/bash

#SBATCH --job-name=LightGCN_optuna
#SBATCH --output=logs/LightGCN_%A_%a.log
#SBATCH --error=logs/LightGCN_%A_%a.err
#SBATCH --array=66
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --exclude=cn-043 


MODEL="lightgcn"
OPTUNA_PARAMS="lightgcn"
NUM_TRIALS=600
N_STARTUP_TRIALS=50
N_JOBS=1
MULTIVARIATE=0
export HYDRA_FULL_ERROR=1

python3 offline_download.py

python3 sbatch/run.py \
  --model "$MODEL" \
  --optuna_params "$OPTUNA_PARAMS" \
  --num_trials "$NUM_TRIALS" \
  --n_startup_trials "$N_STARTUP_TRIALS" \
  --n_jobs "$N_JOBS" \
  $( [ "$MULTIVARIATE" -eq 1 ] && echo "--multivariate" )
