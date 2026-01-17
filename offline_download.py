import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path

from src.utils.download import download


def main():
    parser = argparse.ArgumentParser(description="Download RecSys datasets.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download datasets even if they already exist.",
    )
    args = parser.parse_args()

    dataset_dir = Path("configs/dataset")
    total = sum(1 for _ in dataset_dir.iterdir())
    for dataset_cfg in tqdm(dataset_dir.iterdir(), desc="Downloading datasets...", total=total):
        yaml = OmegaConf.load(dataset_cfg)
        download(yaml.get("url"), f"data/{yaml.get('name')}", force_download=args.force_download)


if __name__ == "__main__":
    main()
