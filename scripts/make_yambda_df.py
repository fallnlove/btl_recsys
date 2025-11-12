import sys
from pathlib import Path

sys.path.append("../../src")

import click
import pandas as pd
from datasets import load_dataset


@click.command()
@click.argument("dataset_type", default="flat/50m")
def main(dataset_type):
    ds = load_dataset(
        "yandex/yambda",
        data_dir=dataset_type,
        data_files="likes.parquet",
        cache_dir="data/cache",
    )

    df = ds["train"].to_pandas()

    new_df = df[["uid", "item_id", "timestamp"]].copy()
    new_df.columns = ["user_id", "item_id", "timestamp"]
    new_df["rating"] = 1
    new_df = new_df[["user_id", "item_id", "rating", "timestamp"]]

    new_df["user_id"], _ = pd.factorize(new_df["user_id"])
    new_df["item_id"], _ = pd.factorize(new_df["item_id"])
    new_df["user_id"] += 1
    new_df["item_id"] += 1

    new_df = new_df.reset_index()
    df_sorted = new_df.sort_values(by=["timestamp", "index"])
    df_sorted = df_sorted.drop(columns=["index"])
    df_sorted = df_sorted.reset_index(drop=True)

    print(f"Final df shape: {new_df.shape}")
    print(f"First few rows:\n{new_df.head()}")

    print("saving csv...")
    folder = Path("data")
    folder.mkdir(exist_ok=True)
    dataset_path = folder / f"yambda_{dataset_type.split('/')[-1]}.csv"
    print(f"{dataset_path=}")
    df_sorted.to_csv(dataset_path, index=False)
    print("CSV saved successfully!")


if __name__ == "__main__":
    main()
