import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def split_by_time(data, user_col, timestamp_col, quantile):
    # Filter interactions
    df = data.groupby(by="user_id").count()["item_id"] >= 3
    good_users = df[df].index
    data = data[data[user_col].isin(good_users)]
    data = data.reset_index(drop=True)

    time_threshold = data[timestamp_col].quantile(quantile)
    train = data[data[timestamp_col] <= time_threshold]
    test = data[data[timestamp_col] > time_threshold]

    return train, test, time_threshold


def split_validation_by_user(train, user_col, validation_size, random_state):
    """Fixed number of users in validation"""
    if validation_size is None:
        raise ValueError(
            "You must specify validation_size parameter for by_user splitting"
        )
    np.random.seed(random_state)
    validation_users = np.random.choice(
        train[user_col].unique(), size=validation_size, replace=False
    )
    validation = train[train[user_col].isin(validation_users)]
    train = train[~train[user_col].isin(validation_users)]
    return train, validation


def split_validation_last_train(train, user_col, timestamp_col):
    train_len = len(train)
    assert train["timestamp"].is_monotonic_increasing, "Train is not monotonic"
    validation = train.groupby(by="user_id").last().reset_index()
    # Drop validation
    train = train[~train.apply(tuple, 1).isin(validation.apply(tuple, 1))]

    assert train_len == len(train) + len(validation)

    return train, validation


def main(
    data_path,
    user_col="user_id",
    item_col="item_id",
    timestamp_col="timestamp",
    train_quantile=0.8,
    validation_type="last",
    val_quantile=0.9,
    validation_size=None,
    random_state=42,
):
    data_path = Path(data_path)
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)

    # Validate required columns
    required_cols = [user_col, timestamp_col]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(
                f"Column '{col}' not found in data. Available columns: {data.columns.tolist()}"
            )

    train, test, train_threshold = split_by_time(
        data, user_col, timestamp_col, train_quantile
    )

    if validation_type == "by_user":
        train, validation = split_validation_by_user(
            train, user_col, validation_size, random_state
        )
    elif validation_type == "by_time":
        train, validation, val_time_threshold = split_by_time(
            train, user_col, timestamp_col, val_quantile
        )
        assert validation[
            "timestamp"
        ].is_monotonic_increasing, "Validation is not monotonic"
    elif validation_type == "last":
        train, validation = split_validation_last_train(train, user_col, timestamp_col)
    else:
        raise ValueError(
            f"Unknown validation_type: {validation_type}. Use 'by_user', 'by_time', or 'last'."
        )

    output_dir = f"data/{data_path.stem}"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    validation_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")

    assert train["timestamp"].is_monotonic_increasing, "Train is not monotonic"
    assert test["timestamp"].is_monotonic_increasing, "Test is not monotonic"
    # assert validation["timestamp"].is_monotonic_increasing, "Validation is not monotonic"

    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)

    print("\nSplit complete!")
    print(f"Train: {len(train)} interactions, {train[user_col].nunique()} users")
    print(
        f"Val: {len(validation)} interactions, {validation[user_col].nunique()} users"
    )
    print(f"Test: {len(test)} interactions, {test[user_col].nunique()} users")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split user-item interaction data into train, validation, and test sets based on timestamps."
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
        "--train_quantile",
        type=float,
        default=0.8,
        help="Quantile for train/test split (default: 0.8)",
    )

    parser.add_argument(
        "--validation_type",
        type=str,
        default="by_time",
        choices=["by_user", "by_time", "last"],
        help="Method to create validation set from train (default: last)",
    )

    parser.add_argument(
        "--val_quantile",
        type=float,
        default=0.9,
        help="Quantile for validation split when using by_time method (default: 0.9)",
    )

    parser.add_argument(
        "--validation_size",
        type=int,
        default=None,
        help="Number of users for validation when using by_user method",
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
        train_quantile=args.train_quantile,
        validation_type=args.validation_type,
        val_quantile=args.val_quantile,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
