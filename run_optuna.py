import json
from copy import deepcopy
from pathlib import Path

import click
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from run_mrgsrec import run_train as run_mrgsrec
from run_sasrec import run_train as run_sasrec


def update_cfg(base_cfg: DictConfig, trial: optuna.trial.Trial) -> DictConfig:
    cfg = deepcopy(base_cfg)

    cfg.model.embedding_dim = trial.suggest_categorical(
        "embedding_dim", [8, 16, 32, 64, 128]
    )
    cfg.model.dropout = trial.suggest_float("dropout", 0.0, 0.6)
    # cfg.optimizer.optimizer.lr = trial.suggest_float("lr", 0.05, 0.1)

    return cfg


def save_trial_results(
    outdir: Path, trial_id: int, params: dict, result, index_path: str
) -> Path:
    trial_name = f"trial-{trial_id:04d}"
    results_path = outdir / f"results_{trial_name}.json"
    payload = {
        "params": params,
        "ndcg@10": result,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    update_index(index_path, trial_id, params, result, results_path)


def update_index(
    index_path: Path, trial_id: int, params: dict, result, results_path: Path
):
    trial_name = f"trial-{trial_id:04d}"

    row = {"trial": trial_name, "results_path": str(results_path)}
    row.update(params | {"ndcg@10": result})

    if index_path.exists():
        df = pd.read_csv(index_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    param_cols = [c for c in df.columns if c not in ("trial", "results_path")]
    df = df[["trial"] + sorted(param_cols) + ["results_path"]]

    df.to_csv(index_path, index=False)


@click.command()
@click.option("--config_path", "-cp", type=str)
@click.option("--num_trials", "-nt", type=int)
@click.option("--exp_name", "-en", type=str)
@click.option("--model", "-m", type=str)
def main(config_path, num_trials, exp_name, model):
    if model == "sasrec":
        run_train = run_sasrec
    elif model == "mrgsrec":
        run_train = run_mrgsrec
    else:
        assert False, "wrong model"
    base_cfg = OmegaConf.load(config_path)
    base_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    outdir = Path("optuna_outputs") / exp_name
    outdir.mkdir(exist_ok=False)
    index_path = outdir / "index.csv"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(base_cfg.get("seed", 42))),
    )

    def objective_fn(trial: optuna.trial.Trial) -> float:
        cfg = update_cfg(base_cfg, trial)

        result = run_train(cfg)

        save_trial_results(outdir, trial.number, trial.params, result, index_path)

        return result

    study.optimize(objective_fn, n_trials=num_trials)

    best = study.best_trial
    print("Best trial:", f"trial-{best.number:04d}", "objective:", best.value)
    print("Best params:", json.dumps(best.params, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
