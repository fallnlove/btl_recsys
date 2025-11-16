import json
from copy import deepcopy
from pathlib import Path

import click
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from hydra import compose, initialize

from run_model import run_train
from src.utils import create_logger

logger = create_logger(name=__name__)


def update_cfg(base_cfg: DictConfig, trial: optuna.trial.Trial) -> DictConfig:
    cfg = deepcopy(base_cfg)

    # cfg.model.embedding_dim = trial.suggest_categorical(
    #     "embedding_dim", [16, 32, 64, 128]
    # )
    # cfg.model.num_layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4])
    # cfg.model.num_heads = trial.suggest_categorical("num_heads", [1, 2, 4])
    # cfg.model.dim_feedforward = trial.suggest_categorical(
    #     "dim_feedforward", [32, 64, 128, 256]
    # )
    # cfg.model.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    # cfg.optimizer.optimizer.lr = trial.suggest_loguniform("lr", 1e-4, 3e-3)

    if cfg.model_name == "mrgsrec":
        cfg.model.num_hops = trial.suggest_categorical("num_hops", [1, 2, 3])
        cfg.model.eta = trial.suggest_float("eta", 0.5, 1.0)

        cfg.loss.local_coef = trial.suggest_float("local_coef", 0.0, 1.0)
        cfg.loss.global_coef = trial.suggest_float("global_coef", 0.0, 1.0)
        cfg.loss.fusion_coef = trial.suggest_float("fusion_coef", 0.0, 1.0)
        cfg.loss.contrastive_coef = trial.suggest_float("contrastive_coef", 0.0, 1.0)

    logger.info(f"Trial params: {trial.params}")

    return cfg


def save_trial_results(
    outdir: Path, trial_id: int, params: dict, metrics, index_path: str
) -> Path:
    trial_name = f"trial-{trial_id:04d}"
    results_path = outdir / f"results_{trial_name}.json"
    payload = {"params": params} | metrics
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    update_index(index_path, trial_id, params, metrics, results_path)


def update_index(
    index_path: Path, trial_id: int, params: dict, metrics, results_path: Path
):
    trial_name = f"trial-{trial_id:04d}"

    row = {"trial": trial_name, "results_path": str(results_path)}
    row.update(params | metrics)

    if index_path.exists():
        df = pd.read_csv(index_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    param_cols = [c for c in df.columns if c not in ("trial", "results_path")]
    df = df[["trial"] + sorted(param_cols) + ["results_path"]]

    df.to_csv(index_path, index=False)


@click.command()
@click.option("--config_name", "-cp", type=str)
@click.option("--num_trials", "-nt", type=int)
@click.option("--exp_name", "-en", type=str)
@click.option("--model_name", "-mn", type=str)
@click.option("--parallel_mode", "-pm", is_flag=True, default=False)
@click.option("--dataset_name", "-ds", default=None)
def main(config_name, num_trials, exp_name, model_name, parallel_mode, dataset_name):
    with initialize(config_path="configs"):
        base_cfg = compose(config_name=config_name)
    OmegaConf.to_yaml(base_cfg)
    OmegaConf.set_struct(base_cfg, False)
    base_cfg["model_name"] = model_name
    if dataset_name is not None:
        base_cfg["dataset"]["name"] = dataset_name

    outdir = Path("optuna_outputs") / exp_name
    if parallel_mode:
        outdir.mkdir(exist_ok=True)
    else:
        outdir.mkdir(exist_ok=False)
    index_path = outdir / "index.csv"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=100),
        study_name=exp_name,
        storage=JournalStorage(
            JournalFileBackend(file_path=str(outdir / f"./{exp_name}.log"))
        ),
        load_if_exists=True,
    )

    def objective_fn(trial: optuna.trial.Trial) -> float:
        cfg = update_cfg(base_cfg, trial)

        _, metrics = run_train(cfg)

        save_trial_results(outdir, trial.number, trial.params, metrics, index_path)

        return metrics["val/ndcg@10"]

    study.optimize(objective_fn, n_trials=num_trials)

    best = study.best_trial
    print("Best trial:", f"trial-{best.number:04d}", "objective:", best.value)
    print("Best params:", json.dumps(best.params, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
