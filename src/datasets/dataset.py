import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_array

from torch.utils.data import Dataset

class RecSysDataset(Dataset):
    def __init__(self, name: str, split: str = "train"):
        assert split in ["train", "val", "test"], "Split must be one of 'train', 'val', or 'test'."
        self._split = split
        self.name = name

        self.all_path = Path("data") / (name + ".csv")
        self.train_path = Path("data/global_split") / name / "train.csv"
        self.val_path = Path("data/global_split") / name / "validation.csv"
        self.test_path = Path("data/global_split") / name / "test.csv"

        all_df = pd.read_csv(self.all_path)
        self._df_train = pd.read_csv(self.train_path).sort_values(by=["timestamp"])
        self._df_val = pd.read_csv(self.val_path).sort_values(by=["timestamp"])
        self._df_test = pd.read_csv(self.test_path).sort_values(by=["timestamp"])

        self._n_users = all_df["user_id"].nunique()
        self._n_items = all_df["item_id"].nunique()

        if split == "train":
            self._df = self._df_train
        elif split == "val":
            self._df = self._df_val
            holdout_path = Path("data/global_split") / name / "holdout_validation.csv"
            self._holdout_df = pd.read_csv(holdout_path)
            self._holdout = np.zeros(self.n_users, dtype=np.long)
            self._holdout[self._holdout_df['user_id'].values] = self._holdout_df['item_id'].values
        elif split == "test":
            self._df = self._df_test
            holdout_path = Path("data/global_split") / name / "holdout_test.csv"
            self._holdout_df = pd.read_csv(holdout_path)
            self._holdout = np.zeros(self.n_users, dtype=np.long)
            self._holdout[self._holdout_df['user_id'].values] = self._holdout_df['item_id'].values

        self._users = self._df["user_id"].unique()
        self._index = self._create_index()
    
    def _create_index(self):
        if self._split == "train":
            return [
                torch.Tensor(self._df[self._df['user_id'] == user_id]['item_id'].tolist())
                for user_id in self._users
            ]

        df_merged = pd.concat(
            [self._df_train, self._df_val]
            if self._split == "val" else
            [self._df_train, self._df_val, self._df_test],
            ignore_index=True
        ).sort_values(by=["timestamp"])

        return [
            torch.Tensor(df_merged[df_merged['user_id'] == user_id]
                .loc[lambda x: x['timestamp'] < self._holdout_df[self._holdout_df['user_id'] == user_id]['timestamp'].item()]
                ['item_id'].tolist())
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
