#!/bin/bash

#SBATCH --job-name=ml_optuna
#SBATCH --error=logs/ml_optuna-%A-%a.err
#SBATCH --output=logs/ml_optuna-%A-%a.log
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-4

module load Python
module load gnu14
conda activate mrgsrec

python run_optuna.py --config_name ml1m_mrgsrec --num_trials 10000 --exp_name ml1m_mrgsrec --model_name mrgsrec --parallel_mode --dataset_name ml1m
