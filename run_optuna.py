import shutil
from copy import deepcopy
from pathlib import Path

import click
import optuna
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial

from run_model import run_train
from src.utils import save_metrics

CONFIG_DIR = "configs"
OPTUNA_DIR = "optuna_outputs"
TARGET_METRIC = "ndcg@10"


def suggest_cfg(config: DictConfig, trial: Trial) -> DictConfig:
    new_config = deepcopy(config)

    for param in config.optuna_params:
        name = param.name
        type_ = param.type

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


class Objective:
    def __init__(
        self,
        config: DictConfig,
        artifact_store: FileSystemArtifactStore,
        tmp_dir: Path,
    ) -> None:
        self._config = config
        self._tmp_dir = tmp_dir
        self._artifact_store = artifact_store

    def __call__(self, trial: Trial) -> float:
        suggested_config = suggest_cfg(self._config, trial)

        best_metrics = run_train(suggested_config)

        files_dir = self._tmp_dir / str(trial.number)
        files_dir.mkdir(exist_ok=False)

        metric_keys = set(best_metrics.keys())
        for key in metric_keys:
            trial.set_user_attr(key, best_metrics[key])

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

        return best_metrics[f"val/{TARGET_METRIC}"]


@click.command()
@click.option("--config_name", "-cn", type=str)
@click.option("--dataset", "-ds", type=str)
@click.option("--optuna_params", "-op", type=str)
@click.option("--experiment_name", "-en", type=str)
@click.option("--timeout", "-to", type=float, default=4 * 60 * 60)
@click.option("--num_trials", "-nt", default=None, type=int)
@click.option("--verbose", "-v", is_flag=True, default=True)
def main(
    config_name: str,
    dataset: str,
    optuna_params: str,
    experiment_name: str,
    timeout: float,
    num_trials: int,
    verbose: bool,
):
    out_dir = Path(OPTUNA_DIR) / experiment_name
    tmp_dir = out_dir / "tmp"
    artifact_dir = out_dir / "artifacts"

    out_dir.mkdir(exist_ok=False, parents=True)
    tmp_dir.mkdir(exist_ok=False, parents=True)
    artifact_dir.mkdir(exist_ok=False, parents=True)

    artifact_store = FileSystemArtifactStore(base_path=str(artifact_dir))

    with initialize(config_path=CONFIG_DIR):
        base_cfg = compose(config_name=config_name, overrides=[
            f"+optuna_params={optuna_params}",
            f"dataset={dataset}"
        ])

    OmegaConf.set_struct(base_cfg, False)

    if verbose:
        print(OmegaConf.to_yaml(base_cfg))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=100),
        study_name=experiment_name,
        storage=JournalStorage(
            JournalFileBackend(file_path=str(out_dir / f"./{experiment_name}.log"))
        ),
        load_if_exists=True,
    )

    study.optimize(
        Objective(config=base_cfg, artifact_store=artifact_store, tmp_dir=tmp_dir),
        n_trials=num_trials,
        timeout=timeout,
    )

    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
