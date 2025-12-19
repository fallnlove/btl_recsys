#!/bin/bash

#SBATCH --job-name=lgcn_ciao
#SBATCH --error=logs/lgcn_ciao-%A-%a.err
#SBATCH --output=logs/lgcn_ciao-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

python run_optuna.py --config_name lightgcn --dataset ciao_dvd --optuna_params lightgcn_medium --experiment_name lightgcn_ciao_dvd
