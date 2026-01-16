import click
import pandas as pd
import numpy as np

from pathlib import Path


@click.command()
@click.option("--filename", type=str)
@click.option("--user_col", type=str, default="user_id")
@click.option("--item_col", type=str, default="item_id")
@click.option("--time_col", type=str, default=None)
@click.option("--rating_col", type=str, default=None)
def main(
    filename: str,
    user_col: str,
    item_col: str,
    time_col: str,
    rating_col: str,
):
    df = pd.read_csv(
        filename,
        engine="python",
    )
    if time_col is None or time_col not in df.columns:
        df["timestamp"] = np.random.randint(1, 1_000_000_000, size=len(df))
        time_col = "timestamp"
        
    if rating_col is not None:
        df["rating"] = 1
        rating_col = "rating"

    df["user_id"], _ = pd.factorize(df[user_col])
    df["item_id"], _ = pd.factorize(df[item_col])
    df["timestamp"] = df[time_col]
    # from 2020-07-23 01:42:04 to seconds
    if df['timestamp'].dtype == object:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').astype(
            np.int64
        ) // 10**9
    df['rating'] = df[rating_col]
    df = df[['user_id', 'item_id', 'timestamp', 'rating']].copy()

    df = df.reset_index()
    df_sorted = df.sort_values(by=[time_col, "index"])
    df_sorted = df_sorted.drop(columns=["index"])
    df_sorted = df_sorted.reset_index(drop=True)

    df_sorted.to_csv(f"data/{Path(filename).stem}.csv", index=False)

if __name__ == "__main__":
    main()
