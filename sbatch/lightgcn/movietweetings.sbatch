#!/bin/bash

#SBATCH --job-name=lgcn_tweets
#SBATCH --error=logs/lgcn_tweets-%A-%a.err
#SBATCH --output=logs/lgcn_tweets-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

python run_optuna.py --config_name lightgcn --dataset movietweetings --optuna_params lightgcn_medium --experiment_name lightgcn_movietweetings
