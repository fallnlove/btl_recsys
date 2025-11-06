import copy
import json
import time
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from src.dataset import SequenceDataset, build_graph
from src.loss import MRGSRecLoss
from src.metrics import BaseMetric, NDCGMetric, StatefullMetric
from src.model import MRGSRecModel
from src.optimizer import BasicOptimizer
from src.utils import (BasicBatchProcessor, create_logger, fix_random_seed,
                       inference, train)

logger = create_logger(name=__name__)
seed_val = 42


def unpack_dataset(dataset_config):
    data_folder = Path(dataset_config["path_to_data_dir"])
    dataset_name = dataset_config["name"]
    all_data = pd.read_csv(data_folder / f"{dataset_name}.csv")
    split_folder = data_folder / "global_split" / dataset_name
    train_path = split_folder / "train.csv"
    val_path = split_folder / "validation.csv"
    test_path = split_folder / "test.csv"

    return all_data, train_path, val_path, test_path


@hydra.main(version_base=None, config_path="configs", config_name="ml1m")
def main(cfg):
    run_train(cfg)


def run_train(cfg):
    fix_random_seed(seed_val)
    config = OmegaConf.to_container(cfg, resolve=True)

    device = config["device"]

    logger.info(f"Training config:\n{OmegaConf.to_yaml(config)}\n")
    logger.info(f"Current DEVICE: {device}")

    # TODO: dumb a little
    all_data, train_path, val_path, test_path = unpack_dataset(config["dataset"])

    dataset_meta = {
        "num_users": all_data["user_id"].max(),
        "num_items": all_data["item_id"].max(),
        "max_sequence_length": config["dataset"]["max_sequence_length"],
    }

    train_sampler = SequenceDataset(
        train_path,
        config["dataset"]["max_sequence_length"],
        mode="train",
    )

    validation_sampler = SequenceDataset(
        val_path,
        config["dataset"]["max_sequence_length"],
        mode="test",
        all_data=all_data,
    )

    test_sampler = SequenceDataset(
        test_path,
        config["dataset"]["max_sequence_length"],
        mode="test",
    )

    graph = build_graph(
        train_sampler, config["dataset"]["path_to_data_dir"], device, dataset_meta
    )

    collator = BasicBatchProcessor()
    train_dataloader = DataLoader(
        dataset=train_sampler, **config["dataloader"]["train"], collate_fn=collator
    )
    validation_dataloader = DataLoader(
        dataset=validation_sampler,
        **config["dataloader"]["validation"],
        collate_fn=collator,
    )
    test_dataloader = DataLoader(
        dataset=test_sampler, **config["dataloader"]["validation"], collate_fn=collator
    )

    model = MRGSRecModel.create_from_config(
        config["model"], graph=graph, **dataset_meta
    ).to(device)
    loss_function = MRGSRecLoss.create_from_config(config["loss"])
    optimizer = BasicOptimizer.create_from_config(config["optimizer"], model=model)
    optimizer_fi = BasicOptimizer.create_from_config(
        config["optimizer_fi"], model=model
    )

    logger.debug("Everything is ready for training process!")

    metrics = {"ndcg@10": NDCGMetric(10)}

    inference_dict_validation = dict(
        dataloader=validation_dataloader,
        model=model,
        metrics=metrics,
        device=device,
    )

    inference_dict_test = dict(
        dataloader=test_dataloader,
        model=model,
        metrics=metrics,
        device=device,
    )

    # Train process
    metrics = train(
        dataloader=train_dataloader,
        warm_dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        optimizer_fi=optimizer_fi,
        loss_function=loss_function,
        num_epochs=config["num_epochs"],
        early_stopping_rounds=config["early_stopping_rounds"],
        best_metric=config.get("best_metric"),
        inference_dict_validation=inference_dict_validation,
        inference_dict_test=inference_dict_test,
        device=device,
    )
    print(f"val/ndcg@10 = {metrics['val']}")
    print(f"test/ndcg@10 = {metrics['test']}")
    return metrics


if __name__ == "__main__":
    main()
