#!/bin/bash

#SBATCH --job-name=lgcn_ml1m
#SBATCH --error=logs/lgcn_ml1m-%A-%a.err
#SBATCH --output=logs/lgcn_ml1m-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

python run_optuna.py --config_name lightgcn --dataset movielens --optuna_params lightgcn_dense --experiment_name lightgcn_movielens
