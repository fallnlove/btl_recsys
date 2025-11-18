#!/bin/bash

#SBATCH --job-name=sports_optuna
#SBATCH --error=logs/sports-%A-%a.err
#SBATCH --output=logs/sports-%A-%a.log
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-4

module load Python
module load gnu14
conda activate mrgsrec

python run_optuna.py --config_name sports_mrgsrec --num_trials 10000 --exp_name sports_mrgsrec --model_name mrgsrec --parallel_mode --dataset_name reviews_Sports_and_Outdoors_5