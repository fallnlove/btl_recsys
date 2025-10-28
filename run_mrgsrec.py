import copy
import json
import time

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from pathlib import Path
import pandas as pd

from src.dataset import ScientificDataset, build_graph, SequenceDataset
from src.loss import MRGSRecLoss
from src.metrics import BaseMetric, StatefullMetric
from src.model import MRGSRecModel
from src.optimizer import BasicOptimizer
from src.utils import (
    BasicBatchProcessor,
    create_logger,
    fix_random_seed,
    inference,
    train,
)
from src.metrics import NDCGMetric

logger = create_logger(name=__name__)
seed_val = 42


def unpack_dataset(dataset_config):
    data_folder = Path(dataset_config["path_to_data_dir"])
    all_data = pd.read_csv(data_folder / f"{dataset_config['name']}.csv")
    split_folder = data_folder / "global_split"
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

    logger.info("Training config: \n{}".format(OmegaConf.to_yaml(config)))
    logger.info("Current DEVICE: {}".format(device))

    # TODO: dumb a little
    all_data, train_path, val_path, test_path = unpack_dataset(config["dataset"])

    dataset_meta = {
        "num_users": all_data["user_id"].max(),
        "num_items": all_data["item_id"].max(),
        "max_sequence_length": config["dataset"]["max_sequence_length"],
    }

    dataset = ScientificDataset(
        config["dataset"]["max_sequence_length"],
        config["dataset"]["path_to_data_dir"],
        config["dataset"]["name"],
    )

    train_index, validation_index, test_index = dataset.get_index()

    train_sampler = SequenceDataset(
        train_index,
        mode="train",
    )

    validation_sampler = SequenceDataset(
        validation_index,
        mode="eval",
    )

    test_sampler = SequenceDataset(
        test_index,
        mode="eval",
    )

    graph = build_graph(
        train_sampler, config["dataset"]["path_to_data_dir"], device, dataset_meta
    )

    collator = BasicBatchProcessor()
    train_dataloader = DataLoader(
        dataset=train_sampler, **config["dataloader"]["train"], collate_fn=collator
    )
    warm_dataloader = DataLoader(
        dataset=train_sampler, **config["dataloader"]["warm_val"], collate_fn=collator
    )
    validation_dataloader = DataLoader(
        dataset=validation_sampler,
        **config["dataloader"]["validation"],
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
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

    inference_dict = dict(
        dataloader=validation_dataloader,
        model=model,
        metrics=metrics,
        device=device,
    )

    # Train process
    best_metric = train(
        dataloader=train_dataloader,
        warm_dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        optimizer_fi=optimizer_fi,
        loss_function=loss_function,
        num_epochs=config.get("num_epochs", 100),
        early_stopping_rounds=config.get("early_stopping_rounds", 10),
        best_metric=config.get("best_metric"),
        inference_dict=inference_dict,
        device=device,
    )
    print(f"ndcg@10 = {best_metric}")
    return best_metric


if __name__ == "__main__":
    main()
