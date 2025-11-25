# Multi-Representation Recommender Systems

```
.
└── RecSys-BTL/
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

#### Amazon

Datasets were taken from the [Amazon Product Data repository](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) (5-core).

To unpack an Amazon dataset and convert it into the required format, run:

```bash
python3 scripts/make_amazon_df.py <path_to_raw_data_file>
```

#### MovieLens1M

Download ML-1M from [source](https://files.grouplens.org/datasets/movielens/), move `ratings.dat` to the `raw_data` folder, and then run:

```bash
python3 scripts/make_ml1m_df.py
```

#### Validation

You can validate your dataset and view basic statistics with:

```bash
python3 scripts/check_datasets.py <path_to_csv>
```

## Splits

To create a global time-point split, run:

```bash
python3 scripts/global_split.py --data_path=<path_to_csv>
```

For a detailed description of all available options:

```bash
python3 scripts/global_split.py --help
```

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

### Models

Two types of models are available: **SASRec** and **MRGSRec**.

To train SASRec, set `model_name=sasrec` in the config, or specify it via CLI:

```bash
python3 run_model.py -cn <your_config> model_name=sasrec
```

Same for MRGSRec.

## Hyperparameter Optimization

To run hyperparameter optimization, use the `run_optuna.py` script:

```bash
python3 run_optuna.py --config_path <path_to_config>
--num_trials <number_of_trials>
--exp_name <experiment_name>
--model <model_name>
--dataset_name <dataset_name>
```

Arguments:

* `--config_name` (`-cn`): name of the base configuration file used for the experiments  
* `--num_trials` (`-nt`): number of optimization trials  
* `--exp_name` (`-en`): experiment name (used for naming the output folder)  
* `--model_name` (`-mn`): model type (`sasrec` or `mrgsrec`)

## Results Management

All optimization results are stored in the `optuna_outputs/` directory.  
For each experiment, a subfolder named `<exp_name>` is created.

Inside this folder:

* **`index.csv`** — a summary table containing:
  * trial identifiers (e.g., `trial-0001`)  
  * paths to the corresponding JSON files  
  * hyperparameter values  
  * final evaluation metrics (e.g., `ndcg@10`)

* **Per-trial JSON files** — each file stores the complete set of hyperparameters and the resulting metrics for that specific trial.
