import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main(
    data_path,
    random_states=[],
    user_col="user_id",
    item_col="item_id",
    timestamp_col="timestamp",
    holdout_type="custom",
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
    elif holdout_type == "custom":
        validation = data_val.groupby(user_col).tail(n=1).reset_index(drop=True)
        tests = [data_test.groupby(user_col).sample(n=1, random_state=random_state).reset_index(drop=True) for random_state in random_states]
    else:
        raise ValueError(f"Unknown holdout type: {holdout_type}")

    validation_path = data_path / "holdout_validation.csv"

    validation.to_csv(validation_path, index=False)
    if holdout_type != "custom":
        test_path = data_path / "holdout_test_0.csv"
        test.to_csv(test_path, index=False)
    else:
        for i, _test in enumerate(tests):
            _test.to_csv(data_path / f"holdout_test_{i}.csv", index=False)

    print("\Holdout complete!")
    print(
        f"Val: {len(validation)} holdout items, {validation[user_col].nunique()} users"
    )
    if holdout_type != "custom":
        print(
            f"Test: {len(test)} holdout items, {test[user_col].nunique()} users"
        )
    else:
        for i, _test in enumerate(tests):
            print(
                f"Test {i}: {len(_test)} holdout items, {_test[user_col].nunique()} users"
            )
    

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
        default="custom",
        choices=["custom", "first", "last"],
        help="Type of holdout strategy: custom / first / last (default: custom)",
    )

    parser.add_argument(
        "--random_states",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        help="List of random seeds (default: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51])",
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        user_col=args.user_col,
        item_col=args.item_col,
        timestamp_col=args.timestamp_col,
        holdout_type=args.holdout_type,
        random_states=args.random_states,
    )
