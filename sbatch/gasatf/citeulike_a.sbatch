#!/bin/bash

#SBATCH --job-name=gasatf_citeulike_a
#SBATCH --error=logs/gasatf_citeulike_a-%A-%a.err
#SBATCH --output=logs/gasatf_citeulike_a-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --constraint=type_d

python run_optuna.py --config_name gasatf --dataset citeulike_a --optuna_params gasatf_medium --experiment_name gasatf_citeulike_a
