#!/bin/bash

#SBATCH --job-name=lgcn_groc
#SBATCH --error=logs/lgcn_groc-%A-%a.err
#SBATCH --output=logs/lgcn_groc-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

python run_optuna.py --config_name lightgcn --dataset amazon_ratings_grocery_and_gourmet_food --optuna_params lightgcn_sparse --experiment_name lightgcn_amazon_ratings_grocery_and_gourmet_food
