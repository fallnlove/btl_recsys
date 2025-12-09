import json
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.metrics import (
    Summarizer,
)
from src.utils import (
    create_logger,
    fix_random_seed,
    save_metrics,
)

logger = create_logger(name=__name__)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    metrics = run_train(cfg)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    save_metrics(metrics, output_dir)


def run_train(cfg, verbose: bool = True):
    fix_random_seed(cfg.get("seed", 42))

    config = OmegaConf.to_container(cfg, resolve=True)
    if verbose:
        logger.info(f"Training config:\n{OmegaConf.to_yaml(config)}\n")

    model = instantiate(cfg.model)
    if verbose:
        logger.info(f"Model init: {str(model)}\n")

    train_dataset = instantiate(cfg.dataset, split="train")
    val_dataset = instantiate(cfg.dataset, split="val")

    metrics = Summarizer([
        instantiate(metric_cfg, n_items=train_dataset.n_items) for metric_cfg in cfg.metrics
    ])

    logger.info(f"Start training...\n")
    model.fit(train_dataset, val_dataset)
    predictions = model.predict(val_dataset, top_n=cfg.max_top_n)

    holdout_users = val_dataset.get_holdout_users()
    return metrics(predictions[holdout_users,:], val_dataset.get_holdout_array()[holdout_users])

if __name__ == "__main__":
    main()
