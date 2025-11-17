#!/bin/bash

#SBATCH --job-name=single
#SBATCH --error=logs/single-%A-%a.err
#SBATCH --output=logs/single-%A-%a.log
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

module load Python
module load gnu14
conda activate mrgsrec

python run_model.py -cn beauty_sasrec