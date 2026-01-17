import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_datasets_from_metadata(metadata_path: Path) -> list[str]:
    if not metadata_path.exists():
        raise RuntimeError(
            "metadata.json not found. Run scripts/build_metadata.py first."
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "sorted_configs" in payload and isinstance(payload["sorted_configs"], list):
            return payload["sorted_configs"]
        if "datasets" in payload and isinstance(payload["datasets"], list):
            datasets = payload["datasets"]
        else:
            datasets = []
    elif isinstance(payload, list):
        datasets = payload
    else:
        datasets = []

    if datasets and isinstance(datasets[0], dict):
        datasets = sorted(
            datasets,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        return [item["config"] for item in datasets if "config" in item]

    return [item for item in datasets if isinstance(item, str)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_idx",
        type=int,
        default=None,
        help="1-based dataset index. If omitted, uses SLURM_ARRAY_TASK_ID",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g. random, svd, als, etc.)",
    )
    parser.add_argument(
        "--optuna_params",
        type=str,
        default="empty",
        help="Optuna params config name (from configs/optuna_params)",
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=20,
        help="Number of Optuna startup trials",
    )
    parser.add_argument(
        "--multivariate",
        action="store_true",
        help="Enable Optuna multivariate TPE sampler",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of Optuna jobs to run in parallel",
    )
    parser.add_argument("--num_trials", type=int, default=200)

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]

    print("base_dir =", base_dir)

    metadata_path = base_dir / "data" / "metadata.json"
    if not metadata_path.exists():
        build_script = base_dir / "scripts" / "build_metadata.py"

        print("build_script =", build_script)
        subprocess.run([sys.executable, str(build_script)], check=True)

    datasets = load_datasets_from_metadata(metadata_path)

    if not datasets:
        raise RuntimeError("No datasets found in data/metadata.json")

    dataset_idx = args.dataset_idx
    if dataset_idx is None:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is None:
            raise ValueError(
                "dataset_idx is required when SLURM_ARRAY_TASK_ID is not set"
            )
        dataset_idx = int(task_id)

    print("dataset_idx (1-based) =", dataset_idx)
    dataset_idx = int(dataset_idx) - 1

    if dataset_idx < 0 or dataset_idx >= len(datasets):
        raise ValueError(
            f"dataset_idx must be in [1, {len(datasets)}], "
            f"got {dataset_idx + 1}"
        )

    dataset = datasets[dataset_idx]

    cmd = [
        "python3",
        "run_optuna.py",
        "--config_name", args.model,
        "--dataset", dataset,
        "--optuna_params", args.optuna_params,
        "--experiment_name", args.model + "_" + dataset,
        "--num_trials", str(args.num_trials),
        "--n_startup_trials", str(args.n_startup_trials),
        "--n_jobs", str(args.n_jobs),
        "--multivariate" if args.multivariate else None,
        "--timeout", str(4800 * 60 * 60),
    ]

    cmd = [arg for arg in cmd if arg is not None]

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
