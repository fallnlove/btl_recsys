import json
from pathlib import Path

import hydra
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


def run_train(cfg, verbose: bool = True, test_mode: bool = False):
    fix_random_seed(cfg.get("seed", 42))

    config = OmegaConf.to_container(cfg, resolve=True)
    if verbose:
        logger.info(f"Training config:\n{OmegaConf.to_yaml(config)}\n")

    model = instantiate(cfg.model)
    if verbose:
        logger.info(f"Model init: {str(model)}\n")

    train_dataset = instantiate(cfg.dataset, split="train", merge_train_val=test_mode)
    eval_dataset = instantiate(cfg.dataset, split="test" if test_mode else "val", merge_train_val=test_mode)

    metrics = Summarizer([
        instantiate(metric_cfg, n_items=train_dataset.n_items) for metric_cfg in cfg.metrics
    ])

    if verbose:
        logger.info(f"Start training...\n")
    model.fit(train_dataset, eval_dataset)
    predictions = model.predict(eval_dataset, top_n=cfg.max_top_n)
    if verbose:
        logger.info(f"Training completed.\n")

    holdout_users = eval_dataset.get_holdout_users()
    eval_metrics = {f"{eval_dataset._split}/{k}": v for k, v in metrics(
        predictions[holdout_users,:],
        eval_dataset.get_holdout_array()[holdout_users]
    ).items()}
    if verbose:
        logger.info(f"Final metrics: {json.dumps(eval_metrics, indent=2)}\n")

    return eval_metrics

if __name__ == "__main__":
    main()
