# Multi-Representation Recommender Systems

```
RecSys-BTL/
├── src/
│   ├── base.py
│   ├── model/
│   │   ├── sasrec.py
│   │   ├── ultra_gcn.py
│   │   ├── funk_svd.py
│   │   └── ...
│   ├── metrics/
│   │   ├── hitrate.py
│   │   ├── ndcg.py
│   │   ├── recall.py
│   │   └── coverage.py
│   └── datasets/
│       ├── sequential.py
│       └── non_sequential.py
├── srcipts
├── configs
├── train.py
├── train_optuna.py
├── inference.py
├── btl.py
└── ...
```

## Datasets

To make global time split and random holdout use this:

```bash
python3 scripts/dataset_pipeline.py --filename <path_to_csv>
--user_col <name_of_userid_column>
--item_col <name_of_itemid_column>
--time_col <name_of_timestamp_column>
--rating_col <name_of_rating_column>
```

### parse_datasets_yd

Use this script to download raw CSVs from Yandex Disk, build splits, and generate dataset configs:

```bash
python3 parse_datasets_yd.py --splits_public_url <public_folder_with_splits>
```

After the pipeline finishes, upload the resulting split folders to disk and make the folder public. Then pass that public folder URL via `--splits_public_url` so the generated configs point to the correct location.

## Training

To train a model with a specific configuration, use the following command.  
The configuration file should be placed inside the `configs/` folder.

```bash
python3 run_model.py -cn <your_config>
```

You can also override individual variables directly from the command line:

```bash
python3 run_model.py -cn <your_config> var1=32 var2=32
```

## Hyperparameter Optimization

To run hyperparameter optimization, use the `run_optuna.py` script:

Before running experiments, make sure datasets are downloaded:

```bash
python3 offline_download.py
```

```bash
python3 run_optuna.py --config_name <config_name> \
	--dataset <dataset_name> \
	--optuna_params <optuna_config> \
	--experiment_name <experiment_name>
```

Arguments:

* `--config_name` (`-cn`): name of the base configuration file used for the experiments
* `--dataset` (`-ds`): name of dataset to train on
* `--optuna_params` (`-op`): name of optuna config (see configs/optuna_params)
* `--experiment_name` (`-en`): name of your experiment (should be unique)

## Batch runs (sbatch) and run script

The wrapper script [sbatch/run.py](sbatch/run.py) runs `run_optuna.py` and selects a dataset by index from `data/metadata.json`.

**Submit sbatch jobs from the project root (the folder that contains `run_optuna.py`).**

Make sure to run `offline_download.py` before launching experiments (examples below already include it).

The script calls `offline_download.py`, reads `data/<dataset>/info.json`, sorts by $median_{interactions} \times num_{users}$, and writes the list to `data/metadata.json`.

### Local run

```bash
python3 sbatch/run.py --dataset_idx 1 --model <model_name> --optuna_params <optuna_params> --num_trials 200
```

### Using sbatch arrays

In a SLURM array, the index is taken from `SLURM_ARRAY_TASK_ID` and is expected to be 1-based.

Example sbatch snippet:

```bash
#SBATCH --array=1-10

python3 sbatch/run.py \
	--model <model_name> \
	--optuna_params <optuna_params> \
	--num_trials 200
```

If `data/metadata.json` is missing, `run.py` will automatically call `scripts/build_metadata.py`.

### Sbatch examples

Popular/Random baseline (see [sbatch/popular_random/run_popular_random.sbatch](sbatch/popular_random/run_popular_random.sbatch)):

```bash
#!/bin/bash

#SBATCH --job-name=popular_random_optuna
#SBATCH --output=logs/popular_random_%A_%a.log
#SBATCH --error=logs/popular_random_%A_%a.err
#SBATCH --array=1-93
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1

MODEL="default"
OPTUNA_PARAMS="empty"
NUM_TRIALS=200
N_STARTUP_TRIALS=20
N_JOBS=1
MULTIVARIATE=0
export HYDRA_FULL_ERROR=1

python3 offline_download.py

python3 sbatch/run.py \
	--model "$MODEL" \
	--optuna_params "$OPTUNA_PARAMS" \
	--num_trials "$NUM_TRIALS" \
	--n_startup_trials "$N_STARTUP_TRIALS" \
	--n_jobs "$N_JOBS" \
	$( [ "$MULTIVARIATE" -eq 1 ] && echo "--multivariate" )
```

GASATF (see [sbatch/gasatf/run_gasatf.sbatch](sbatch/gasatf/run_gasatf.sbatch)):

```bash
#!/bin/bash

#SBATCH --job-name=gasatf_optuna
#SBATCH --output=logs/gasatf_%A_%a.log
#SBATCH --error=logs/gasatf_%A_%a.err
#SBATCH --array=1-93
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8

MODEL="gasatf"
OPTUNA_PARAMS="gasatf"
NUM_TRIALS=200
N_STARTUP_TRIALS=20
N_JOBS=4
MULTIVARIATE=0
export HYDRA_FULL_ERROR=1

python3 offline_download.py

python3 sbatch/run.py \
	--model "$MODEL" \
	--optuna_params "$OPTUNA_PARAMS" \
	--num_trials "$NUM_TRIALS" \
	--n_startup_trials "$N_STARTUP_TRIALS" \
	--n_jobs "$N_JOBS" \
	$( [ "$MULTIVARIATE" -eq 1 ] && echo "--multivariate" )
```
