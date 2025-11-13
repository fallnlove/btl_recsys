#!/bin/bash

#SBATCH --job-name=clothes_optuna
#SBATCH --error=logs/clothes_optuna-%A-%a.err
#SBATCH --output=logs/clothes_optuna-%A-%a.log
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-8

module load Python
module load gnu14
conda activate mrgsrec

python run_optuna.py --config_path configs/default.yaml --num_trials 10000 --exp_name clothes --model_name mrgsrec --parallel_mode --dataset_name reviews_Clothing_Shoes_and_Jewelry_5