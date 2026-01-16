import os
import subprocess
import click

from pathlib import Path


@click.command()
@click.option("--filename", type=str)
@click.option("--user_col", type=str, default="user_id")
@click.option("--item_col", type=str, default="item_id")
@click.option("--time_col", type=str, default="timestamp")
@click.option("--rating_col", type=str, default="rating")
def main(filename, user_col, item_col, time_col, rating_col):
    subprocess.run([
        "python",
        "scripts/make_dataset.py",
        "--filename", filename,
        "--user_col", user_col,
        "--item_col", item_col] +
        (["--time_col", time_col] if time_col else []) +
        (["--rating_col", rating_col] if rating_col else [])
    )
    subprocess.run([
        "python",
        "scripts/check_datasets.py",
        f"data/{Path(filename).stem}.csv",
    ])
    subprocess.run([
        "python",
        "scripts/global_split.py",
        "--data_path", f"data/{Path(filename).stem}.csv",
    ])
    subprocess.run([
        "python",
        "scripts/holdout.py",
        "--data_path", f"data/{Path(filename).stem}",
    ])
    os.remove(f"data/{Path(filename).stem}.csv")


if __name__ == "__main__":
    main()
