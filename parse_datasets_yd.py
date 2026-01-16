import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlencode, quote

import requests


PUBLIC_FOLDER_URL = "https://disk.360.yandex.ru/d/9NnCsZwqRtGb8A"
SPLITS_PUBLIC_FOLDER_URL = "error, please set via cmd arg"
TEMP_DATA_DIR = "data_parsed"
CONFIGS_DATASET_DIR = Path("configs") / "dataset"

DEFAULT_COLUMNS = {
    "user_col": "user_id",
    "item_col": "item_id",
    "time_col": "timestamp", 
    "rating_col": "rating"
}


def download_file(direct_url, save_path):
    print(f"Downloading to {save_path}...")
    with requests.get(direct_url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download completed.")


def process_dataset(filename) -> bool:
    print(f"Running pipeline for {filename}...")
    
    command = [
        "python", "scripts/dataset_pipeline.py",
        "--filename", filename,
        "--user_col", DEFAULT_COLUMNS["user_col"],
        "--item_col", DEFAULT_COLUMNS["item_col"]
    ]
    
    if "time_col" in DEFAULT_COLUMNS:
        command.extend(["--time_col", DEFAULT_COLUMNS["time_col"]])
    if "rating_col" in DEFAULT_COLUMNS:
        command.extend(["--rating_col", DEFAULT_COLUMNS["rating_col"]])

    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Pipeline finished successfully for {filename}")
        return True
    print(f"Error processing {filename}:")
    print(result.stderr)
    return False


def build_dataset_url(base_public_url: str, dataset_folder: str) -> str:
    encoded_path = quote(f"/{dataset_folder}", safe="/")
    return f"{base_public_url}?path={encoded_path}"


def write_dataset_config(dataset_name: str, url: str) -> None:
    CONFIGS_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_DATASET_DIR / f"{dataset_name}.yaml"
    content = "\n".join([
        "_target_: src.datasets.RecSysDataset",
        f"name: {dataset_name}",
        f"url: {url}",
        "",
    ])
    config_path.write_text(content)
    print(f"Config written: {config_path}")


def list_public_csv_items(public_key: str) -> list[dict]:
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources"
    items = []
    offset = 0
    limit = 100
    while True:
        params = {
            "public_key": public_key,
            "limit": limit,
            "offset": offset,
        }
        response = requests.get(f"{base_url}?{urlencode(params)}")
        response.raise_for_status()
        embedded = response.json().get("_embedded", {})
        batch = embedded.get("items", [])
        items.extend([item for item in batch if item.get("type") == "file" and item.get("name", "").endswith(".csv")])
        if len(batch) < limit:
            break
        offset += limit
    return items


def main_pipeline(splits_public_url: str):
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)

    try:
        items = list_public_csv_items(PUBLIC_FOLDER_URL)
    except Exception as e:
        print(f"Yandex API Error: {e}")
        return

    for item in items:
        dataset_name = item.get("name")
        file_download_url = item.get("file")

        if not dataset_name or not file_download_url:
            print("Missing dataset name or download link, skipping.")
            continue

        local_file_path = os.path.join(TEMP_DATA_DIR, dataset_name)

        download_file(file_download_url, local_file_path)
        dataset_folder = Path(dataset_name).stem
        ok = process_dataset(local_file_path)
        if ok:
            dataset_url = build_dataset_url(splits_public_url, dataset_folder)
            write_dataset_config(dataset_folder, dataset_url)

            try:
                os.remove(local_file_path)
            except OSError:
                pass

    print("\nAll tasks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets from Yandex Disk, process splits, and write dataset configs."
    )
    parser.add_argument(
        "--splits_public_url",
        type=str,
        default=SPLITS_PUBLIC_FOLDER_URL,
        help=(
            "Public Yandex Disk folder URL for split datasets. "
            "If omitted, uses YD_SPLITS_PUBLIC_URL or PUBLIC_FOLDER_URL."
        ),
    )
    args = parser.parse_args()

    main_pipeline(args.splits_public_url)
