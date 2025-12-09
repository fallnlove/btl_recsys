import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main(
    data_path,
    user_col="user_id",
    item_col="item_id",
    timestamp_col="timestamp",
    holdout_type="random",
    random_state=42,
):
    data_path = Path(data_path)
    print(f"Loading data from: {data_path}")
    data_val = pd.read_csv(data_path / "validation.csv").sort_values(by=[timestamp_col])
    data_test = pd.read_csv(data_path / "test.csv").sort_values(by=[timestamp_col])

    if holdout_type == "first":
        validation = data_val.groupby(user_col).first().reset_index()
        test = data_test.groupby(user_col).first().reset_index()
    elif holdout_type == "last":
        validation = data_val.groupby(user_col).last().reset_index()
        test = data_test.groupby(user_col).last().reset_index()
    elif holdout_type == "random":
        validation = data_val.groupby(user_col).sample(n=1, random_state=random_state).reset_index(drop=True)
        test = data_test.groupby(user_col).sample(n=1, random_state=random_state).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown holdout type: {holdout_type}")

    validation_path = data_path / "holdout_validation.csv"
    test_path = data_path / "holdout_test.csv"

    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)

    print("\Holdout complete!")
    print(
        f"Val: {len(validation)} holdout items, {validation[user_col].nunique()} users"
    )
    print(f"Test: {len(test)} holdout items, {test[user_col].nunique()} users")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to create holdout splits from user-item interaction data"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input CSV file with user-item interactions",
    )

    parser.add_argument(
        "--user_col",
        type=str,
        default="user_id",
        help="Name of user column (default: user_id)",
    )

    parser.add_argument(
        "--item_col",
        type=str,
        default="item_id",
        help="Name of item column (default: item_id)",
    )

    parser.add_argument(
        "--timestamp_col",
        type=str,
        default="timestamp",
        help="Name of timestamp column (default: timestamp)",
    )

    parser.add_argument(
        "--holdout_type",
        type=str,
        default="random",
        choices=["random", "first", "last"],
        help="Type of holdout strategy: random / first / last (default: random)",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        user_col=args.user_col,
        item_col=args.item_col,
        timestamp_col=args.timestamp_col,
        holdout_type=args.holdout_type,
        random_state=args.random_state,
    )
