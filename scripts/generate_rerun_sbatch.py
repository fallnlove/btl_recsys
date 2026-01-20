#!/usr/bin/env python3
"""Generate SLURM sbatch script for re-running incomplete experiments."""
from pathlib import Path
from typing import Dict, List, Tuple
import json
import click


def count_trials_from_log(log_path: Path) -> int:
    """Count unique trial IDs from Optuna log file."""
    trial_ids = set()
    try:
        with log_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                    if "trial_id" in record:
                        trial_ids.add(record["trial_id"])
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return len(trial_ids)


def load_datasets_from_metadata(metadata_path: Path) -> List[str]:
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


def find_incomplete_experiments(optuna_dir: Path) -> List[Tuple[str, int, str, str]]:
    """Find all incomplete experiments and extract model/dataset."""
    incomplete = []
    
    for exp_dir in sorted(optuna_dir.glob("*_*")):
        if not exp_dir.is_dir():
            continue
        
        best_trial_path = exp_dir / "best_trial.yaml"
        if best_trial_path.exists():
            continue
        
        exp_name = exp_dir.name
        parts = exp_name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        
        model, dataset = parts
        
        log_files = list(exp_dir.glob("*.log"))
        n_trials = count_trials_from_log(log_files[0]) if log_files else 0
        
        incomplete.append((exp_name, n_trials, model, dataset))
    
    return incomplete


def generate_sbatch_script(
    incomplete_exps: List[Tuple[str, int, str, str]],
    dataset_to_idx: Dict[str, int],
    output_path: Path,
    num_trials: int = 200,
    n_startup_trials: int = 20,
    n_jobs: int = 1,
    multivariate: bool = False,
) -> None:
    """Generate sbatch script for incomplete experiments using dataset_idx."""

    sbatch_dir = Path("sbatch/incomplete")
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    models = set(model for _, _, model, _ in incomplete_exps)

    for model in sorted(models):
        model_exps = [(exp, trials, m, ds) for exp, trials, m, ds in incomplete_exps if m == model]

        indices = []
        missing = []
        for _, _, _, dataset in model_exps:
            idx = dataset_to_idx.get(dataset)
            if idx is None:
                missing.append(dataset)
            else:
                indices.append(idx)

        if missing:
            click.echo(f"Warning: datasets not found in metadata.json: {missing}")

        if not indices:
            continue

        indices = sorted(set(indices))
        array_expr = ",".join(str(i) for i in indices)

        script_path = sbatch_dir / f"rerun_{model}_incomplete.sbatch"

        with script_path.open("w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"#SBATCH --job-name={model}_incomplete\n")
            f.write(f"#SBATCH --output=logs/{model}_incomplete_%A_%a.log\n")
            f.write(f"#SBATCH --error=logs/{model}_incomplete_%A_%a.err\n")
            f.write(f"#SBATCH --array={array_expr}\n")
            f.write("#SBATCH --time=72:00:00\n")
            f.write("#SBATCH --cpus-per-task=1\n\n")

            f.write(f"MODEL=\"{model}\"\n")
            f.write(f"OPTUNA_PARAMS=\"{model}\"\n")
            f.write(f"NUM_TRIALS={num_trials}\n")
            f.write(f"N_STARTUP_TRIALS={n_startup_trials}\n")
            f.write(f"N_JOBS={n_jobs}\n")
            f.write(f"MULTIVARIATE={1 if multivariate else 0}\n")
            f.write("export HYDRA_FULL_ERROR=1\n\n")
            f.write("python3 offline_download.py\n\n")

            f.write("python3 sbatch/run.py \\\n")
            f.write("  --model \"$MODEL\" \\\n")
            f.write("  --optuna_params \"$OPTUNA_PARAMS\" \\\n")
            f.write("  --num_trials \"$NUM_TRIALS\" \\\n")
            f.write("  --n_startup_trials \"$N_STARTUP_TRIALS\" \\\n")
            f.write("  --n_jobs \"$N_JOBS\" \\\n")
            if multivariate:
                f.write("  --multivariate \\\n")
            f.write("  --dataset_idx \"$SLURM_ARRAY_TASK_ID\"\n")

        script_path.chmod(0o755)

        print(f"Generated: {script_path}")
        print(f"  Models: {model}")
        print(f"  Datasets: {len(indices)}")
        print(f"  SLURM array: {array_expr}")
        print(f"  Command: sbatch {script_path}")
        print()

    print("\nGenerated sbatch files for incomplete experiments.")


@click.command()
@click.option(
    "--num_trials",
    type=int,
    default=200,
    help="Number of trials to run for completion",
)
@click.option(
    "--n_startup_trials",
    type=int,
    default=20,
    help="Number of startup trials",
)
@click.option(
    "--n_jobs",
    type=int,
    default=1,
    help="Number of parallel jobs",
)
@click.option(
    "--multivariate",
    is_flag=True,
    default=False,
    help="Enable multivariate TPE sampler",
)
def main(num_trials: int, n_startup_trials: int, n_jobs: int, multivariate: bool):
    """Generate sbatch scripts for incomplete experiments."""
    optuna_dir = Path("optuna_outputs")
    metadata_path = Path("data") / "metadata.json"
    datasets = load_datasets_from_metadata(metadata_path)
    dataset_to_idx = {name: idx + 1 for idx, name in enumerate(datasets)}
    
    incomplete = find_incomplete_experiments(optuna_dir)
    
    if not incomplete:
        click.echo("All experiments are complete!")
        return
    
    click.echo(f"Found {len(incomplete)} incomplete experiments:\n")
    click.echo(f"{'Experiment':<50} {'Trials':<10}")
    click.echo("-" * 60)
    for exp, n_trials, model, dataset in incomplete:
        click.echo(f"{exp:<50} {n_trials:<10}")
    
    click.echo("\n" + "=" * 60)
    generate_sbatch_script(
        incomplete,
        dataset_to_idx,
        Path("sbatch/incomplete"),
        num_trials=num_trials,
        n_startup_trials=n_startup_trials,
        n_jobs=n_jobs,
        multivariate=multivariate,
    )


if __name__ == "__main__":
    main()
