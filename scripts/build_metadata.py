import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


def run_offline_download(base_dir: Path) -> None:
    subprocess.run([sys.executable, str(base_dir / "offline_download.py")], check=True)


def build_metadata(base_dir: Path) -> dict:
    data_dir = base_dir / "data"
    dataset_cfg_dir = base_dir / "configs" / "dataset"

    datasets: list[dict] = []
    for cfg_path in sorted(dataset_cfg_dir.glob("*.yaml")):
        try:
            cfg = OmegaConf.load(cfg_path)
        except Exception:
            continue

        dataset_name = cfg.get("name") if cfg is not None else None
        if not dataset_name:
            continue

        info_path = data_dir / dataset_name / "info.json"
        if not info_path.exists():
            continue

        try:
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            continue

        median_interactions = info.get("median_interactions")
        num_users = info.get("num_users")
        if median_interactions is None or num_users is None:
            continue

        score = float(median_interactions) * float(num_users)
        datasets.append(
            {
                "config": cfg_path.stem,
                "name": dataset_name,
                "median_interactions": float(median_interactions),
                "num_users": int(num_users),
                "score": score,
            }
        )

    datasets.sort(key=lambda item: item["score"])
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sorted_configs": [item["config"] for item in datasets],
        "datasets": datasets,
        "sort_key": "median_interactions * num_users",
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    run_offline_download(base_dir)

    metadata = build_metadata(base_dir)
    metadata_path = base_dir / "data" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
