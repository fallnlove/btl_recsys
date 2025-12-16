import numpy as np
from scipy import sparse
from typing import Dict
import optuna
import os
import json
import pickle

from src.base import BaseModel

class BPR_MF(BaseModel):
    """
    BPF_MR class.
    """
    rank: int
    n_epochs: int
    learning_rate: float
    regularization: float
    rng: np.random.default_rng
    neg_items: Dict
    P: np.ndarray
    Q: np.ndarray

    def __init__(self, rank, n_epochs, learning_rate, regularization, seed, name: str = "BPR MF"):
        super().__init__(name)
        self.rng = np.random.default_rng(seed)
        self.rank = rank
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

    def fit(self, train_dataset, val_dataset):
        """
        Fit the model to the dataset.
        """
        n_users = train_dataset.n_users
        n_items = train_dataset.n_items

        coo = train_dataset.get_coo_array().astype(np.float32)
        interactions = sparse.csr_matrix((coo.data, (coo.coords)), shape=coo.shape)
        neg_items = {}
        for user in range(n_users):
            neg_items[user] = np.setdiff1d(np.arange(n_items), interactions[user].indices)
        self.neg_items = neg_items

        useridx, itemidx = coo.coords

        shape = n_users, n_items

        self._sgd_sweeps(
            useridx, itemidx, shape, folding_in=False
        )

    def predict(self, dataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.
        """
        n_users = dataset.n_users
        n_items = dataset.n_items
        shape = n_users, n_items

        predictions = np.zeros((dataset.n_users, top_n), dtype=np.int64)
        dataloader = dataset.get_dataloader(batch_size=128, shuffle=False)
        for batch in dataloader:
            batch_users = batch['user_id'].numpy()
            batch_history = batch['history'].numpy().astype(np.int32)
            batch_interactions = np.zeros((len(batch_users), dataset.n_items), dtype=np.float32)

            mask = batch_history != -1
            rows = np.repeat(np.arange(batch_history.shape[0]), mask.sum(axis=1))
            cols = batch_history[mask].ravel()
            batch_interactions[rows, cols] = 1

            for i, user in enumerate(batch_users):
                self.neg_items[user] = np.setdiff1d(self.neg_items[user], batch_history[i][batch_history[i] != -1])

            useridx = np.repeat(batch_users, mask.sum(axis=1))
            itemidx = cols
            self._sgd_sweeps(
                useridx, itemidx, shape, folding_in=True
            )

            batch_scores = self.P[batch_users] @ self.Q.T

            mask = batch_interactions > 0
            batch_scores[mask] = -np.inf

            idx = np.argpartition(-batch_scores, top_n - 1, axis=1)[:, :top_n]
            top_indices_batch = idx[np.arange(len(idx))[:, None],
            np.argsort(-batch_scores[np.arange(len(idx))[:, None], idx], axis=1)]

            predictions[batch_users] = top_indices_batch

        return predictions

    def _sgd_sweeps(
            self, useridx, itemidx, shape, folding_in=False
    ):
        n_users, n_items = shape

        if not folding_in:
            self.P = self.rng.normal(0, np.sqrt(1 / self.rank), (n_users, self.rank))
            self.Q = self.rng.normal(0, np.sqrt(1 / self.rank), (n_items, self.rank))

        for epoch in range(self.n_epochs):
            self._sgd_epoch(
                useridx, itemidx, folding_in
            )

    def _sgd_epoch(
            self, useridx, itemidx, folding_in
    ):
        n_interactions = len(useridx)
        events = self.rng.permutation(n_interactions)

        for eventid in events:
            user = useridx[eventid]
            item = itemidx[eventid]

            j = self.rng.choice(self.neg_items[user])

            pu = self.P[user]
            qi = self.Q[item]
            qj = self.Q[j]

            x_uij = np.dot(pu, qi) - np.dot(pu, qj)

            z = 1 / (1 + np.exp(x_uij))

            grad_u = z * (qi - qj) - self.regularization * pu

            if not folding_in:
                grad_i = z * pu - self.regularization * qi
                grad_j = -z * pu - self.regularization * qj

            pu += self.learning_rate * grad_u
            self.P[user] = pu

            if not folding_in:
                qi += self.learning_rate * grad_i
                qj += self.learning_rate * grad_j
                self.Q[item] = qi
                self.Q[j] = qj

    def save_checkpoint(self, path: str):
        """
        Save the model checkpoint to the specified path.
        """
        os.makedirs(path, exist_ok=True)

        meta = {
            "rank": self.rank,
            "P": "P.npy",
            "Q": "Q.npy",
            "neg_items": "neg_items.pkl"
        }

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

        np.save(os.path.join(path, "P.npy"), self.P)
        np.save(os.path.join(path, "Q.npy"), self.Q)

        with open(os.path.join(path, "neg_items.pkl"), "wb") as f:
            pickle.dump(self.neg_items, f)

    def load_checkpoint(self, path: str):
        """
        Load the model checkpoint from the specified path.
        """
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        self.rank = int(meta["rank"])
        self.P = np.load(os.path.join(path, meta["P"]))
        self.Q = np.load(os.path.join(path, meta["Q"]))
        with open(os.path.join(path, meta["neg_items"]), "rb") as f:
            pickle.load(f)

    def sample_params(self, trial: optuna.trial.Trial):
        """
        Sample hyperparameters for the model using the given trial.
        """
        params = {
            'rank': trial.suggest_int(
                name='rank',
                low=10,
                high=100,
                step=5
            ),
            'n_epochs': trial.suggest_int(
                name='n_epochs',
                low=5,
                high=20,
                step=5
            ),
            'learning_rate': trial.suggest_float(
                name='learning_rate',
                low=1e-4,
                high=1e-1,
                log=True
            ),
            'regularization': trial.suggest_float(
                name='regularization',
                low=1e-5,
                high=1e-1,
                log=True
            )
        }
        return params