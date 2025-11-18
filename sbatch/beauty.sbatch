#!/bin/bash

#SBATCH --job-name=beauty_optuna
#SBATCH --error=logs/beauty_optuna-%A-%a.err
#SBATCH --output=logs/beauty_optuna-%A-%a.log
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-4

module load Python
module load gnu14
conda activate mrgsrec

python run_optuna.py --config_name beauty_mrgsrec --num_trials 10000 --exp_name beauty_mrgsrec --model_name mrgsrec --parallel_mode --dataset_name reviews_Beauty_5