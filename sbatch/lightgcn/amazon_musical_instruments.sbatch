#!/bin/bash

#SBATCH --job-name=lgcn_music
#SBATCH --error=logs/lgcn_music-%A-%a.err
#SBATCH --output=logs/lgcn_music-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

python run_optuna.py --config_name lightgcn --dataset amazon_musical_instruments --optuna_params lightgcn_sparse --experiment_name lightgcn_amazon_musical_instruments
