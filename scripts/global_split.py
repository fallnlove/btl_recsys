import pandas as pd
import numpy as np
import argparse
import os

def split_by_time(data, user_col, timestamp_col, quantile):
    data = data.sort_values([user_col, timestamp_col], kind="stable")
    
    time_threshold = data[timestamp_col].quantile(quantile)
    user_second_timestamp = data.groupby(user_col)[timestamp_col].nth(1)
    train_users = user_second_timestamp[
        user_second_timestamp <= time_threshold
    ].index
    train = data[data[user_col].isin(train_users)]
    train = train[train[timestamp_col] <= time_threshold]
    user_last_timestamp = data.groupby(user_col)[timestamp_col].nth(-1)
    test_users = user_last_timestamp[user_last_timestamp > time_threshold].index
    test = data[data[user_col].isin(test_users)]
    return train, test, time_threshold

def split_validation_by_user(train, user_col, validation_size, random_state):
    """Fixed number of users in validation"""
    if validation_size is None:
        raise ValueError("You must specify validation_size parameter for by_user splitting")
    np.random.seed(random_state)
    validation_users = np.random.choice(
        train[user_col].unique(), size=validation_size, replace=False
    )
    validation = train[train[user_col].isin(validation_users)]
    train = train[~train[user_col].isin(validation_users)]
    return train, validation

def split_validation_last_train(train, user_col, timestamp_col):
    train = train.sort_values(
        [user_col, timestamp_col], kind="stable")
    train["time_idx_reversed"] = train.groupby(user_col).cumcount(ascending=False)
    
    validation = train[
        (train['time_idx_reversed'] == 0) &
        (train.groupby(user_col)['time_idx_reversed'].transform('max') >= 1)  # last interaction for users with 2+ interactions
    ].drop(columns=["time_idx_reversed"])
    
    train = train[
        (train['time_idx_reversed'] >= 1) &
        (train.groupby(user_col)['time_idx_reversed'].transform('max') >= 1)  # all but last interaction for users with 2+ interactions
    ].drop(columns=["time_idx_reversed"])
    
    return train, validation

def main(data_path, user_col='user_id', item_col='item_id',
         timestamp_col='timestamp', train_quantile=0.8, 
         validation_type='last', val_quantile=0.9,
         validation_size=None, random_state=42):
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    
    # Validate required columns
    required_cols = [user_col, timestamp_col]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data. Available columns: {data.columns.tolist()}")
    

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
    elif validation_type == "last":
        train, validation = split_validation_last_train(
            train, user_col, timestamp_col
        )
    else:
        raise ValueError(f"Unknown validation_type: {validation_type}. Use 'by_user', 'by_time', or 'last'.")
    
    output_dir = '../data2/preprocessed'
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    validation_path = os.path.join(output_dir, 'validation.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"\nSplit complete!")
    print(f"Train: {len(train)} interactions, {train[user_col].nunique()} users")
    print(f"Val: {len(validation)} interactions, {validation[user_col].nunique()} users")
    print(f"Test: {len(test)} interactions, {test[user_col].nunique()} users")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split user-item interaction data into train, validation, and test sets based on timestamps.'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to input CSV file with user-item interactions'
    )
    
    parser.add_argument(
        '--user_col',
        type=str,
        default='user_id',
        help='Name of user column (default: user_id)'
    )
    
    parser.add_argument(
        '--item_col',
        type=str,
        default='item_id',
        help='Name of item column (default: item_id)'
    )
    
    parser.add_argument(
        '--timestamp_col',
        type=str,
        default='timestamp',
        help='Name of timestamp column (default: timestamp)'
    )
    
    parser.add_argument(
        '--train_quantile',
        type=float,
        default=0.8,
        help='Quantile for train/test split (default: 0.8)'
    )
    
    parser.add_argument(
        '--validation_type',
        type=str,
        default='last',
        choices=['by_user', 'by_time', 'last'],
        help='Method to create validation set from train (default: last)'
    )
    
    parser.add_argument(
        '--val_quantile',
        type=float,
        default=0.9,
        help='Quantile for validation split when using by_time method (default: 0.9)'
    )
    
    parser.add_argument(
        '--validation_size',
        type=int,
        default=None,
        help='Number of users for validation when using by_user method'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
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
        random_state=args.random_state
    )
