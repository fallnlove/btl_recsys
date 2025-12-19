#!/bin/bash

#SBATCH --job-name=mf_sgd_optuna
#SBATCH --error=logs/mf_sgd_optuna-%A-%a.err
#SBATCH --output=logs/mf_sgd_optuna-%A-%a.log
#SBATCH --time=9:10:00
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --constraint=type_d
 
python run_optuna.py --config_name mf_sgd --dataset amazon_toy_and_games --optuna_params mf_sgd_sparse_datasets --experiment_name mf_sgd_amazon_toy_and_games
