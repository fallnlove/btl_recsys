import json
import os

import optuna
from optuna.storages import JournalFileStorage, JournalStorage


def is_false(value):
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return value.strip().lower() == "false"
    return False


def log_has_downvote_false(log_path):
    try:
        storage = JournalStorage(JournalFileStorage(log_path))
        studies = storage.get_all_studies()
        if not studies:
            return False
        study = optuna.load_study(study_name=studies[0].study_name, storage=storage)
    except Exception:
        return False

    try:
        trial = study.best_trial
    except Exception:
        return False

    params = trial.params or {}
    for key, value in params.items():
        key_lower = str(key).lower()
        if ("downvote" in key_lower or "seen" in key_lower) and is_false(value):
            return True

    return False


def load_sorted_configs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    configs = data.get("sorted_configs", []) if isinstance(data, dict) else []
    if not isinstance(configs, list):
        return {}

    return {str(name): idx + 1 for idx, name in enumerate(configs)}


def extract_dataset_name(algo_name: str, exp_name: str) -> str:
    prefix = f"{algo_name}_"
    if exp_name.startswith(prefix):
        return exp_name[len(prefix) :]
    return exp_name


def main():
    root = os.path.join(os.getcwd(), "optuna_outputs", "BTL_results")  # TODO path
    metadata_path = os.path.join(os.getcwd(), "data", "metadata.json")
    sorted_index = load_sorted_configs(metadata_path)

    for algo_name in sorted(os.listdir(root)):
        algo_path = os.path.join(root, algo_name)
        if not os.path.isdir(algo_path):
            continue
        datasets_with_false = []
        for exp_name in sorted(os.listdir(algo_path)):
            exp_path = os.path.join(algo_path, exp_name)
            if not os.path.isdir(exp_path):
                continue
            log_files = sorted(
                os.path.join(exp_path, name)
                for name in os.listdir(exp_path)
                if name.endswith(".log")
            )
            log_has_false = any(log_has_downvote_false(path) for path in log_files)

            if log_has_false:
                datasets_with_false.append(exp_name)

        if datasets_with_false:
            print('-' * 50)
            print(algo_name)
            dataset_names = [
                extract_dataset_name(algo_name, name)
                for name in sorted(datasets_with_false)
            ]
            indices = [
                str(sorted_index.get(name, "NA"))
                for name in dataset_names
            ]
            print("datasets:", ", ".join(dataset_names))
            print("indices:", ", ".join(indices))
            print('=' * 50)


if __name__ == "__main__":
    main()
