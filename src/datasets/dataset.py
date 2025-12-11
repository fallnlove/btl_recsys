import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_array

from torch.utils.data import Dataset

from src.utils.download import download


class RecSysDataset(Dataset):
    def __init__(self, name: str, url: str, split: str = "train"):
        assert split in ["train", "val", "test"], "Split must be one of 'train', 'val', or 'test'."
        self._split = split
        self.name = name
        folder = Path("data") / name

        download(url, "data/")
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
            self._df_merged = self._df_train
        elif split == "val":
            self._df = self._df_val
            holdout_path = folder / "holdout_validation.csv"
            self._df_merged = pd.concat(
                [self._df_train, self._df_val],
                ignore_index=True
            ).sort_values(by=["timestamp"])
        elif split == "test":
            self._df = self._df_test
            holdout_path = folder / "holdout_test.csv"
            self._df_merged = pd.concat(
                [self._df_train, self._df_val, self._df_test],
                ignore_index=True
            ).sort_values(by=["timestamp"])
        
        if split in ["val", "test"]:
            self._holdout_df = pd.read_csv(holdout_path)
            self._holdout = np.zeros(self.n_users, dtype=np.int64)
            self._holdout[self._holdout_df['user_id'].values] = self._holdout_df['item_id'].values

        self._users = self._df["user_id"].unique()
        self._index = self._create_index()
    
    def _create_index(self):
        if self._split == "train":
            groups = (
                self._df.groupby('user_id')['item_id']
                .apply(list)
                .to_dict()
            )
            return [
                torch.Tensor(groups.get(user_id, []))
                for user_id in self._users
            ]

        holdout_ts = (
            self._holdout_df[['user_id', 'timestamp']]
                .rename(columns={'timestamp': 'holdout_ts'})
        )
        df = self._df_merged.merge(holdout_ts, on='user_id', how='left')
        df = df[df['timestamp'] < df['holdout_ts']]
        df = df.sort_values(['user_id', 'timestamp'])
        groups = (
            df.groupby('user_id')['item_id']
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
        assert self._split == "train", "COO array can only be created for training data."
        return coo_array(
            (np.ones(self._df["user_id"].values.shape[0]), 
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
