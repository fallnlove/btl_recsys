import copy
import json
import time

import hydra
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from src.dataset import ScientificDataset, build_graph
from src.loss import LocalObjective, MRGSRecLoss
from src.metrics import BaseMetric, StatefullMetric
from src.model import MRGSRecModel
from src.optimizer import BasicOptimizer
from src.sequence import SequentialEncoder
from src.utils import (BasicBatchProcessor, create_logger, fix_random_seed,
                       inference, train)

logger = create_logger(name=__name__)
seed_val = 42


@hydra.main(version_base=None, config_path="configs", config_name="ml1m")
def main(cfg):
    run_train(cfg)


def run_train(cfg):
    fix_random_seed(seed_val)
    config = OmegaConf.to_container(cfg, resolve=True)

    device = config["device"]

    logger.info("Training config: \n{}".format(OmegaConf.to_yaml(config)))
    logger.info("Current DEVICE: {}".format(device))

    dataset = ScientificDataset(
        config["dataset"]["max_sequence_length"],
        config["dataset"]["path_to_data_dir"],
        config["dataset"]["name"],
    )

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

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

    _embedding_dim = config["model"]["embedding_dim"]
    _num_users = dataset.meta["num_users"]
    _num_items = dataset.meta["num_items"]
    _max_sequence_length = dataset.meta["max_sequence_length"]

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
    model = SequentialEncoder(
        **config["model"],
        position_embeddings=position_embeddings,
        item_embeddings=item_embeddings,
        num_items=_num_items,
    ).to(device)
    loss_function = LocalObjective("positive")
    optimizer = BasicOptimizer.create_from_config(config["optimizer"], model=model)
    optimizer_fi = BasicOptimizer.create_from_config(
        config["optimizer_fi"], model=model
    )

    logger.debug("Everything is ready for training process!")

    metrics = {
        metric_name: BaseMetric.create_from_config(metric_cfg, **dataset.meta)
        for metric_name, metric_cfg in config["metrics"].items()
    }

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
        num_epochs=config.get("num_epochs", 200),
        early_stopping_rounds=config.get("early_stopping_rounds", 50),
        device=device,
        best_metric=config.get("best_metric"),
        inference_dict=inference_dict,
    )
    print(f"ndcg@10 = {best_metric}")
    return best_metric


if __name__ == "__main__":
    main()
