from pathlib import Path

import click
import pandas as pd

REQUIRED_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


@click.command()
@click.argument("filename")
def main(filename: str):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    df = pd.read_csv(path)

    # 1. Columns check
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(
                f"Not enough columns: {df.columns.tolist()} " f"expected: {column}"
            )

    # 2. user_id / item_id check
    for col in ["user_id", "item_id"]:
        if df[col].min() != 0 or df[col].nunique() != df[col].max() + 1:
            raise ValueError(f"{col} must be consecutive starting from 0")

    # 3. Timestamp sort check
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Dataset must be sorted by timestamp")

    # 4. Unique users and items
    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()

    # 5. average and median interactions length
    interactions_per_user = df["user_id"].value_counts()
    avg_interactions = interactions_per_user.mean()
    median_interactions = interactions_per_user.median()

    # 6. sparsity
    sparsity = len(df) / (num_users * num_items)

    print("✅ Dataset is valid!")
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Avg interactions per user: {avg_interactions:.4f}")
    print(f"Median interactions per user: {median_interactions:.4f}")
    print(f"Sparsity: {sparsity:.8f}")

    (path.parent / Path(filename).stem).mkdir(exist_ok=True)
    with open(path.parent / Path(filename).stem / f"info.json", "w") as f:
        info = {
            "num_users": num_users,
            "num_items": num_items,
            "avg_interactions": avg_interactions,
            "median_interactions": median_interactions,
            "sparsity": sparsity,
        }
        import json

        json.dump(info, f, indent=4)


if __name__ == "__main__":
    main()
