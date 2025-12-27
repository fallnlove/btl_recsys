#!/bin/bash

#SBATCH --job-name=gasatf_movietweetings
#SBATCH --error=logs/gasatf_movietweetings-%A-%a.err
#SBATCH --output=logs/gasatf_movietweetings-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --constraint=type_d

python run_optuna.py --config_name gasatf --dataset movietweetings --optuna_params gasatf_medium --experiment_name gasatf_movietweetings
