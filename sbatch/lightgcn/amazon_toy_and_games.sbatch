#!/bin/bash

#SBATCH --job-name=lgcn_toys
#SBATCH --error=logs/lgcn_toys-%A-%a.err
#SBATCH --output=logs/lgcn_toys-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

python run_optuna.py --config_name lightgcn --dataset amazon_toy_and_games --optuna_params lightgcn_sparse --experiment_name lightgcn_amazon_toy_and_games
