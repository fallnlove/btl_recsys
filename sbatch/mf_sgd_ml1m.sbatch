#!/bin/bash

#SBATCH --job-name=ml_optuna
#SBATCH --error=logs/ml_optuna-%A-%a.err
#SBATCH --output=logs/ml_optuna-%A-%a.log
#SBATCH --time=4:10:00
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8

python run_optuna.py --config_name mf_sgd --dataset movielens --optuna_params mf_sgd_ml1m --experiment_name mf_sgd_ml1m
