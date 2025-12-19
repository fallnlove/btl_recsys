#!/bin/bash

#SBATCH --job-name=lgcn_anime
#SBATCH --error=logs/lgcn_anime-%A-%a.err
#SBATCH --output=logs/lgcn_anime-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

python run_optuna.py --config_name lightgcn --dataset anime_ratings --optuna_params lightgcn_dense --experiment_name lightgcn_anime_ratings
