#!/bin/bash

#SBATCH --job-name=ml_optuna
#SBATCH --error=logs/ml_optuna-%A-%a.err
#SBATCH --output=logs/ml_optuna-%A-%a.log
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-8

module load Python
module load gnu14
conda activate mrgsrec

python run_optuna.py --config_path configs/default.yaml --num_trials 10000 --exp_name ml --model_name mrgsrec --parallel_mode --dataset_name ml1m
