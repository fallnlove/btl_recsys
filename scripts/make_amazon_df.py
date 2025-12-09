import argparse
import gzip
from pathlib import Path

import click
import numpy as np
import pandas as pd


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


@click.command()
@click.argument("filename")
def main(filename):
    filename = Path(filename)
    ds_name = filename.name.split(".")[0]
    print(f"{ds_name=}")
    df = getDF(filename)

    new_df = df[["user_id", "asin", "rating", "timestamp"]].copy()
    new_df.columns = ["user_id", "item_id", "rating", "timestamp"]
    new_df["rating"] = 1

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
    dataset_path = folder / f"{ds_name}.csv"
    print(f"{dataset_path=}")
    df_sorted.to_csv(dataset_path, index=False)
    print("CSV saved successfully!")


if __name__ == "__main__":
    main()
