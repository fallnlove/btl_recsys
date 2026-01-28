import shutil
from copy import deepcopy
import json
import hashlib
from pathlib import Path

import click
import optuna
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial, TrialState

from run_model import run_train
from src.utils import save_metrics

import logging

# Mute Numba debug logs
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.core.ssa').setLevel(logging.WARNING)
logging.getLogger('numba.core.byteflow').setLevel(logging.WARNING)

CONFIG_DIR = "configs"
OPTUNA_DIR = "optuna_outputs"
TARGET_METRIC = "ndcg@10"


def suggest_cfg(config: DictConfig, trial: Trial) -> DictConfig:
    new_config = deepcopy(config)

    for param in config.optuna_params:
        name = param.name
        type_ = param.type
        cond = param.get("condition", None)

        if cond is not None:
            cond_name = cond["name"]
            cond_values = cond["values"]
            if cond_name in trial.params and trial.params[cond_name] not in cond_values:
                continue
        
        if type_ == "categorical":
            value = trial.suggest_categorical(name, param.choices)
        elif type_ == "float":
            low = param.low
            high = param.high
            step = param.get("step", None)
            log = param.get("log", False)
            value = trial.suggest_float(name, low, high, step=step, log=log)
        elif type_ == "int":
            low = param.low
            high = param.high
            step = param.get("step", 1)
            log = param.get("log", False)
            value = trial.suggest_int(name, low, high, step=step, log=log)
        else:
            raise ValueError(f"Unknown parameter type: {type_}")
        
        new_config.model.update({name: value})

    return new_config

def set_params(config: DictConfig, params: dict) -> DictConfig:
    new_config = deepcopy(config)

    for name, value in params.items():
        new_config.model.update({name: value})

    return new_config


class Objective:
    def __init__(
        self,
        config: DictConfig,
        artifact_store: FileSystemArtifactStore,
        tmp_dir: Path,
        study: optuna.Study,
    ) -> None:
        self._config = config
        self._tmp_dir = tmp_dir
        self._artifact_store = artifact_store
        self._study = study

    @staticmethod
    def _params_hash(params: dict) -> str:
        payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _find_completed_trial_by_hash(self, params_hash: str) -> Trial | None:
        for existing in self._study.get_trials(deepcopy=False):
            if existing.state == TrialState.COMPLETE:
                if existing.user_attrs.get("config_hash") == params_hash:
                    return existing
        return None

    def __call__(self, trial: Trial) -> float:
        suggested_config = suggest_cfg(self._config, trial)
        params_hash = self._params_hash(trial.params)

        trial.set_user_attr("config_hash", params_hash)
        existing_trial = self._find_completed_trial_by_hash(params_hash)
        if existing_trial:
            cached_metrics = existing_trial.user_attrs.get("cached_metrics")
            if isinstance(cached_metrics, dict):
                for key, value in cached_metrics.items():
                    trial.set_user_attr(key, value)
                metric_key = f"val/{TARGET_METRIC}"
                if metric_key in cached_metrics:
                    trial.set_user_attr("cached", True)
                    trial.set_user_attr("cached_from_trial", existing_trial.number)
                    return cached_metrics[metric_key]
        try:
            best_metrics = run_train(suggested_config, verbose=False, test_mode=False)
        except Exception as exc:
            trial.set_user_attr("error", repr(exc))
            raise optuna.TrialPruned() from exc

        files_dir = self._tmp_dir / str(trial.number)
        if files_dir.exists():
            shutil.rmtree(files_dir)
        files_dir.mkdir(parents=True, exist_ok=True)

        metric_keys = set(best_metrics.keys())
        for key in metric_keys:
            trial.set_user_attr(key, best_metrics[key])
        trial.set_user_attr("cached_metrics", best_metrics)

        save_metrics(best_metrics, output_dir=files_dir)

        for file in files_dir.iterdir():
            if file.is_file() and file.suffix == ".png":
                artifact_id = upload_artifact(
                    artifact_store=self._artifact_store,
                    file_path=str(file.resolve()),
                    study_or_trial=trial,
                )

                trial.set_user_attr(file.stem, artifact_id)

        shutil.rmtree(files_dir)

        metric_key = f"val/{TARGET_METRIC}"
        if metric_key not in best_metrics:
            trial.set_user_attr("error", f"Missing metric: {metric_key}")
            raise optuna.TrialPruned()

        return best_metrics[metric_key]


@click.command()
@click.option("--config_name", "-cn", type=str)
@click.option("--dataset", "-ds", type=str)
@click.option("--optuna_params", "-op", type=str)
@click.option("--experiment_name", "-en", type=str)
@click.option("--timeout", "-to", type=float, default=4800 * 60 * 60)
@click.option("--num_trials", "-nt", default=200, type=int)
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option('--multivariate', '-mv', is_flag=True, default=False)
@click.option('--n_startup_trials', '-nst', default=20, type=int)
@click.option('--n_jobs', '-nj', default=1, type=int)
def main(
    config_name: str,
    dataset: str,
    optuna_params: str,
    experiment_name: str,
    timeout: float,
    num_trials: int,
    verbose: bool,
    multivariate: bool,
    n_startup_trials: int,
    n_jobs: int,
):
    out_dir = Path(OPTUNA_DIR) / experiment_name
    tmp_dir = out_dir / "tmp"
    artifact_dir = out_dir / "artifacts"

    out_dir.mkdir(exist_ok=True, parents=True)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    artifact_dir.mkdir(exist_ok=True, parents=True)

    artifact_store = FileSystemArtifactStore(base_path=str(artifact_dir))

    with initialize(config_path=CONFIG_DIR):
        base_cfg = compose(config_name=config_name, overrides=[
            f"+optuna_params={optuna_params}",
            f"dataset={dataset}"
        ])

    OmegaConf.set_struct(base_cfg, False)

    if verbose:
        print(OmegaConf.to_yaml(base_cfg))

    storage = JournalStorage(
        JournalFileBackend(file_path=str(out_dir / f"./{experiment_name}.log"))
    )

    existing_trials = 0
    for summary in optuna.get_all_study_summaries(storage=storage):
        if summary.study_name == experiment_name:
            existing_trials = summary.n_trials
            break

    remaining_trials = max(0, num_trials - existing_trials)
    remaining_startup_trials = max(0, n_startup_trials - existing_trials)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=remaining_startup_trials,
            multivariate=multivariate,
        ),
        study_name=experiment_name,
        storage=storage,
        load_if_exists=True,
    )

    if remaining_trials > 0:
        study.optimize(
            Objective(
                config=base_cfg,
                artifact_store=artifact_store,
                tmp_dir=tmp_dir,
                study=study,
            ),
            n_trials=remaining_trials,
            timeout=timeout,
            n_jobs=n_jobs,
        )

    best = study.best_trial
    best_cfg = set_params(base_cfg, best.params)

    additional_cfg_params = best.user_attrs.get("additional_cfg_params")
    if isinstance(additional_cfg_params, dict) and additional_cfg_params:
        for name, value in additional_cfg_params.items():
            best_cfg.model.update({name: value})
  
    best_metrics = run_train(best_cfg, verbose=verbose, test_mode=True)
    best_payload = {
        "number": best.number,
        "value": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,
        "test_metrics": best_metrics,
    }

    best_file = out_dir / "best_trial.yaml"
    with open(best_file, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(best_payload)))

    best_config = out_dir / f"{experiment_name}.yaml"
    with open(best_config, "w") as f:
        f.write(OmegaConf.to_yaml(best_cfg))

    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()


if __name__ == "__main__":
    main()
