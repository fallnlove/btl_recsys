import copy
import json
import time
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from src.dataset import SequenceDataset, build_graph
from src.loss import LocalObjective, MRGSRecLoss
from src.metrics import NDCGMetric
from src.model import MRGSRecModel
from src.optimizer import BasicOptimizer
from src.sequence import SequentialEncoder
from src.utils import (
    BasicBatchProcessor,
    create_logger,
    fix_random_seed,
    inference,
    train,
)

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
    model_name = cfg["model_name"]
    assert model_name in ["sasrec", "mrgsrec"]
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

    if model_name == "sasrec":
        _embedding_dim = config["model"]["embedding_dim"]
        _num_users = dataset_meta["num_users"]
        _num_items = dataset_meta["num_items"]
        _max_sequence_length = dataset_meta["max_sequence_length"]

        item_embeddings = nn.Embedding(
            num_embeddings=_num_items + 2,
            embedding_dim=_embedding_dim,
            padding_idx=0,
        )
        position_embeddings = nn.Embedding(
            num_embeddings=_max_sequence_length
            + 1,  # in order to include `max_sequence_length` value
            embedding_dim=_embedding_dim,
        )
        del config["model"]["initializer_range"]
        del config["model"]["num_hops"]
        model = SequentialEncoder(
            **config["model"],
            position_embeddings=position_embeddings,
            item_embeddings=item_embeddings,
            num_items=_num_items,
        ).to(device)
        loss_function = LocalObjective()
        optimizer = BasicOptimizer.create_from_config(config["optimizer"], model=model)
    elif model_name == "mrgsrec":
        graph = build_graph(
            train_sampler, config["dataset"]["path_to_data_dir"], device, dataset_meta
        )
        model = MRGSRecModel(
            cfg=config["model"],
            num_items=dataset_meta["num_items"],
            num_users=dataset_meta["num_users"],
            max_sequence_length=dataset_meta["max_sequence_length"],
            graph=graph,
        )
        loss_function = MRGSRecLoss(config["loss"])
        optimizer = BasicOptimizer.create_from_config(config["optimizer"], model=model)

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
        loss_function=loss_function,
        num_epochs=config["num_epochs"],
        early_stopping_rounds=config["early_stopping_rounds"],
        device=device,
        best_metric=config.get("best_metric"),
        inference_dict_validation=inference_dict_validation,
        inference_dict_test=inference_dict_test,
    )
    print(f"val/ndcg@10 = {metrics['val']}")
    print(f"test/ndcg@10 = {metrics['test']}")
    return metrics


if __name__ == "__main__":
    main()
