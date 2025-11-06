# Multi-Representation Recommender Systems

## Datasets

Datasets were taken from [the Amazon Product Data repository](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) (5-core).

To unpack an Amazon dataset, and then use:

```bash
python3 scripts/make_amazon_df.py <path_to_raw_data_file>
```

You can also validate your dataset and view some statistics with:

```bash
python3 scripts/check_datasets.py <path_to_csv>
```

## Splits

For a global time-point split, use:

```bash
python3 scripts/global_split.py --data_path=<path_to_csv>
```

For a detailed description of available options, run: 

```bash
python3 scripts/global_split.py --help
```

## Training

To train a model with a specific configuration, use the following command.
The configuration file should be placed inside the `configs/` folder.

```bash
python3 run_mrgsrec.py -cn <your_config>
```

You can also override some variables directly from the command line:

```bash
python3 run_mrgsrec.py -cn <your_config> var1=32 var2=32
```

## Models

Two types of models are available: **SASRec** and **MRGSRec**.

* To train SASRec, use `run_sasrec.py`.
* To train MRGSRec, use `run_mrgsrec.py`.

## Hyperparameter Optimization

To run hyperparameter optimization, use the script `run_optuna.py`:

```bash
python3 run_optuna.py --config_path <path_to_config> \
                      --num_trials <number_of_trials> \
                      --exp_name <experiment_name> \
                      --model <model_name>
```

Arguments:

* `--config_path` (`-cp`): path to the base configuration file used for the experiments
* `--num_trials` (`-nt`): number of optimization trials
* `--exp_name` (`-en`): name of the experiment (used for output folder naming)
* `--model` (`-m`): model type (`sasrec` or `mrgsrec`)

### Results Management

All optimization results are stored under the `optuna_outputs/` directory.
For each experiment, a new subfolder named `<exp_name>` is created.
Inside this folder:

* **`index.csv`** — a summary table with all trials, containing

  * trial identifiers (e.g., `trial-0001`)
  * paths to the corresponding JSON files
  * hyperparameter values
  * final evaluation metrics (e.g., `ndcg@10`)

* **Per-trial JSON files** — each file stores the full set of hyperparameters and resulting metrics for that trial.

