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

```bash
python3 run_optuna.py --config_path <path_to_config>
--dataset <dataset_name>
--optuna_params <optuna_config>
--experiment_name <experiment_name>
```

Arguments:

* `--config_name` (`-cn`): name of the base configuration file used for the experiments
* `--dataset` (`-ds`): name of dataset to train on
* `--optuna_params` (`-op`): name of optuna config (see configs/optuna_params)
* `--experiment_name` (`-en`): name of your experiment (should be unique)
