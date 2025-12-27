#!/bin/bash

#SBATCH --job-name=gasatf_ml1m
#SBATCH --error=logs/gasatf_ml1m-%A-%a.err
#SBATCH --output=logs/gasatf_ml1m-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --constraint=type_d

python run_optuna.py --config_name gasatf --dataset movielens --optuna_params gasatf_dense --experiment_name gasatf_movielens
