#!/bin/bash

#SBATCH --job-name=ml_optuna
#SBATCH --error=logs/ml_optuna-%A-%a.err
#SBATCH --output=logs/ml_optuna-%A-%a.log
#SBATCH --time=9:10:00
#SBATCH --nodes=1
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

python run_optuna.py --config_name lightgcn --dataset movielens --optuna_params lightgcn_ml1m --experiment_name lightgcn_ml1m
