import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_array

from torch.utils.data import Dataset

from src.utils.download import download


class RecSysDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 url: str, 
                 split: str, 
                 merge_train_val: bool,
                 holdout_filename: str = ""):
        assert split in ["train", "val", "test"], "Split must be one of 'train', 'val', or 'test'."
        assert not (merge_train_val and split == "val"), "You cannot use validation split when merging train and val sets."
        self._split = split
        self._merge_train_val = merge_train_val
        self.name = name
        folder = Path("data") / name

        download(url, f"data/{name}")
        info = folder / "info.json"
        train_path = folder / "train.csv"
        val_path = folder / "validation.csv"
        test_path = folder / "test.csv"

        self.meta_info = json.load(open(info, "r"))
        self._df_train = pd.read_csv(train_path).sort_values(by=["timestamp"])
        self._df_val = pd.read_csv(val_path).sort_values(by=["timestamp"])
        self._df_test = pd.read_csv(test_path).sort_values(by=["timestamp"])

        self._n_users = self.meta_info["num_users"]
        self._n_items = self.meta_info["num_items"]

        if split == "train":
            self._df = self._df_train
        elif split == "val":
            holdout_path = folder / holdout_filename
            self._df = pd.concat(
                [self._df_train, self._df_val],
                ignore_index=True
            ).sort_values(by=["timestamp"])
        elif split == "test":
            holdout_path = folder / holdout_filename
            self._df = pd.concat(
                [self._df_train, self._df_val, self._df_test],
                ignore_index=True
            ).sort_values(by=["timestamp"])
        
        if split in ["val", "test"]:
            self._holdout_df = pd.read_csv(holdout_path)
            self.delete_holdout_items()
            self._holdout = np.zeros(self.n_users, dtype=np.int64)
            self._holdout[self._holdout_df['user_id'].values] = self._holdout_df['item_id'].values

        if merge_train_val and split == "train":
            self._df = pd.concat(
                [self._df_train, self._df_val],
                ignore_index=True
            ).sort_values(by=["timestamp"])
        self._users = self._df["user_id"].unique()
        self._index = self._create_index()

    def delete_holdout_items(self):
        holdout_ts = (
            self._holdout_df[['user_id', 'timestamp']]
                .rename(columns={'timestamp': 'holdout_ts'})
        )
        df = self._df.merge(holdout_ts, on='user_id', how='left')
        df = df[df['timestamp'] < df['holdout_ts']]
        df = df.sort_values(['timestamp'])
        self._df = df.reset_index(drop=True)
        # delete users with no history
        users_with_history = self._df['user_id'].unique()
        self._holdout_df = self._holdout_df[
            self._holdout_df['user_id'].isin(users_with_history)
        ].reset_index(drop=True)

    def _create_index(self):
        groups = (
            self._df.groupby('user_id')['item_id']
            .apply(list)
            .to_dict()
        )
        return [
            torch.tensor(groups.get(user_id, []))
            for user_id in self._users
        ]

    @property
    def n_users(self) -> int:
        return self._n_users

    @property
    def n_items(self) -> int:
        return self._n_items

    def get_coo_array(self) -> coo_array:
        return coo_array(
            (np.ones(self._df["user_id"].values.shape[0]),
             (self._df["user_id"].values, self._df["item_id"].values)),
            shape=(self.n_users, self.n_items)
        )

    def get_coo_array_rating(self) -> coo_array:
        return coo_array(
             (self._df["rating"].values,
             (self._df["user_id"].values, self._df["item_id"].values)),
            shape=(self.n_users, self.n_items)
        )

    def get_holdout_array(self) -> np.ndarray:
        assert self._split in ["val", "test"], "Holdout array can only be created for validation or test data."
        return self._holdout

    def get_holdout_users(self) -> np.ndarray:
        assert self._split in ["val", "test"], "Holdout items can only be retrieved for validation or test data."
        return self._holdout_df['user_id'].values

    def __len__(self):
        return len(self._users)

    def __getitem__(self, idx):
        result = {
            "user_id": self._users[idx],
            "history": self._index[idx],
        }
        if self._split in ["val", "test"]:
            result["holdout"] = self._holdout[self._users[idx]]
        return result

    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        from torch.utils.data import DataLoader
        from src.utils.collate import collate_fn

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
