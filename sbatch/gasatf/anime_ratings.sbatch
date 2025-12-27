#!/bin/bash

#SBATCH --job-name=gasatf_anime
#SBATCH --error=logs/gasatf_anime-%A-%a.err
#SBATCH --output=logs/gasatf_anime-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --constraint=type_d

python run_optuna.py --config_name gasatf --dataset anime_ratings --optuna_params gasatf_dense --experiment_name gasatf_anime_ratings
