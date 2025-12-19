#!/bin/bash

#SBATCH --job-name=lgcn_cds
#SBATCH --error=logs/lgcn_cds-%A-%a.err
#SBATCH --output=logs/lgcn_cds-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

python run_optuna.py --config_name lightgcn --dataset amazon_ratings_cds_and_vinyl --optuna_params lightgcn_sparse --experiment_name lightgcn_amazon_ratings_cds_and_vinyl
