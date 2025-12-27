#!/bin/bash

#SBATCH --job-name=gasatf_music
#SBATCH --error=logs/gasatf_music-%A-%a.err
#SBATCH --output=logs/gasatf_music-%A-%a.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --constraint=type_d

python run_optuna.py --config_name gasatf --dataset amazon_musical_instruments --optuna_params gasatf_sparse --experiment_name gasatf_amazon_musical_instruments
