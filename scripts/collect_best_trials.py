from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import pandas as pd
import yaml
import click


def load_yaml(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def find_config_file(directory: Path) -> Optional[Path]:
    candidates = [p for p in directory.glob("*.yaml") if p.name != "best_trial.yaml"]
    return sorted(candidates)[0] if candidates else None


def count_trials_from_log(directory: Path) -> Optional[int]:
    log_candidates = sorted(directory.glob("*.log"))
    if not log_candidates:
        return None

    trial_ids = set()
    for log_path in log_candidates:
        try:
            with log_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "trial_id" in record:
                        trial_ids.add(record["trial_id"])
        except OSError:
            continue
    return len(trial_ids) if trial_ids else None


def extract_dataset_model(config: Dict) -> Tuple[Optional[str], Optional[str]]:
    dataset_cfg = config.get("dataset") or {}
    model_cfg = config.get("model") or {}

    dataset = dataset_cfg.get("name") if isinstance(dataset_cfg, dict) else None
    model = None
    if isinstance(model_cfg, dict):
        model = model_cfg.get("name") or model_cfg.get("_target_")
        if model and "." in model and not model_cfg.get("name"):
            model = model.split(".")[-1]
    return dataset, model


def process_directory(directory: Path) -> Optional[Dict[str, object]]:
    best_trial_path = directory / "best_trial.yaml"
    if not best_trial_path.is_file():
        return None

    config_path = find_config_file(directory)
    if not config_path or not config_path.is_file():
        return None

    best_data = load_yaml(best_trial_path)
    config_data = load_yaml(config_path)

    dataset, model = extract_dataset_model(config_data)

    row: Dict[str, object] = {
        "dataset": dataset,
        "model": model,
        "n_trials": count_trials_from_log(directory),
    }

    for key, value in (best_data.get("test_metrics") or {}).items():
        row[key] = value

    return row


def collect_best_trials(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        row = process_directory(entry)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


@click.command(help="Aggregate best trials across Optuna output folders (depth 1).")
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to save the summary files. Defaults to the root folder.",
)
def main(root: Path, output_dir: Optional[Path]):
    output_dir = output_dir or root
    df = collect_best_trials(root)

    if df.empty:
        click.echo("No best_trial.yaml and config YAML pairs were found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "best_trials_summary.csv"
    xlsx_path = output_dir / "best_trials_summary.xlsx"

    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
        click.echo(f"Saved {len(df)} rows to {csv_path} and {xlsx_path}.")
    except ImportError as e:
        click.echo(f"Saved {len(df)} rows to {csv_path}. Excel save failed: {e}. Install openpyxl for Excel support.")


if __name__ == "__main__":
    main()
