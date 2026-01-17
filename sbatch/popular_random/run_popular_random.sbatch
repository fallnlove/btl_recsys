#!/bin/bash

#SBATCH --job-name=popular_random_optuna
#SBATCH --output=logs/popular_random_%A_%a.log
#SBATCH --error=logs/popular_random_%A_%a.err
#SBATCH --array=1-93
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1

MODEL="default"
OPTUNA_PARAMS="empty"
NUM_TRIALS=200
N_STARTUP_TRIALS=20
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
