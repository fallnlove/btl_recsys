import numpy as np
from scipy import sparse
from typing import Dict
import optuna
import os
import json
import pickle
from numba import njit, prange

from src.base import BaseModel


@njit(cache=True)
def sgd_sweeps(
    useridx,
    itemidx,
    n_users,
    n_items,
    rank,
    n_epochs,
    learning_rate,
    regularization,
    P,
    Q,
    indptr,
    indices,
    folding_in,
):
    for _ in range(n_epochs):
        sgd_epoch(
            useridx,
            itemidx,
            n_items,
            rank,
            learning_rate,
            regularization,
            P,
            Q,
            indptr,
            indices,
            folding_in,
        )


@njit(cache=True)
def sgd_epoch(
    useridx,
    itemidx,
    n_items,
    rank,
    learning_rate,
    regularization,
    P,
    Q,
    indptr,
    indices,
    folding_in,
):
    n_interactions = len(useridx)

    for idx in prange(n_interactions):
        user = useridx[idx]
        item = itemidx[idx]

        j = np.random.randint(0, n_items)

        pu = P[user]
        qi = Q[item]
        qj = Q[j]

        x_uij = np.dot(pu, qi) - np.dot(pu, qj)

        z = 1.0 / (1.0 + np.exp(x_uij))

        grad_u = z * (qi - qj) - regularization * pu

        if not folding_in:
            grad_i = z * pu - regularization * qi
            grad_j = -z * pu - regularization * qj

        pu += learning_rate * grad_u
        P[user] = pu

        if not folding_in:
            qi += learning_rate * grad_i
            qj += learning_rate * grad_j
            Q[item] = qi
            Q[j] = qj


class BPR_MF(BaseModel):
    """
    BPF_MR class.
    """
    rank: int
    n_epochs: int
    learning_rate: float
    regularization: float
    neg_items: Dict
    P: np.ndarray
    Q: np.ndarray

    def __init__(self, rank, n_epochs, learning_rate, regularization, seed, name: str = "BPR MF"):
        super().__init__(name)
        self.rank = rank
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.P = None
        self.Q = None

    def fit(self, train_dataset, val_dataset=None):
        n_users = train_dataset.n_users
        n_items = train_dataset.n_items

        coo = train_dataset.get_coo_array().astype(np.float32)
        csr = sparse.csr_matrix(
            (coo.data, (coo.coords)),
            shape=(n_users, n_items)
        )

        useridx, itemidx = coo.coords

        self.P = self.rng.normal(0, np.sqrt(1 / self.rank), (n_users, self.rank))
        self.Q = self.rng.normal(0, np.sqrt(1 / self.rank), (n_items, self.rank))

        sgd_sweeps(
            useridx=useridx.astype(np.int32),
            itemidx=itemidx.astype(np.int32),
            n_users=n_users,
            n_items=n_items,
            rank=self.rank,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            P=self.P,
            Q=self.Q,
            indptr=csr.indptr.astype(np.int32),
            indices=csr.indices.astype(np.int32),
            folding_in=False,
        )

    def predict(self, dataset, top_n: int) -> np.ndarray:
        n_users = dataset.n_users
        n_items = dataset.n_items

        predictions = np.zeros((n_users, top_n), dtype=np.int64)

        dataloader = dataset.get_dataloader(batch_size=128, shuffle=False)

        for batch in dataloader:
            batch_users = batch["user_id"].numpy().astype(np.int32)
            batch_history = batch["history"].numpy().astype(np.int32)

            B, _ = batch_history.shape

            mask = batch_history != -1
            nnz_per_user = mask.sum(axis=1).astype(np.int32)

            indptr = np.zeros(B + 1, dtype=np.int32)
            indptr[1:] = np.cumsum(nnz_per_user)

            indices = batch_history[mask].astype(np.int32)
            itemidx = indices

            rows = np.repeat(np.arange(batch_history.shape[0]), mask.sum(axis=1))
            cols = batch_history[mask]

            useridx = np.repeat(batch_users, mask.sum(axis=1))

            sgd_sweeps(
                useridx=useridx,
                itemidx=itemidx,
                n_users=n_users,
                n_items=n_items,
                rank=self.rank,
                n_epochs=5,
                learning_rate=self.learning_rate,
                regularization=self.regularization,
                P=self.P,
                Q=self.Q,
                indptr=indptr,
                indices=indices,
                folding_in=True
            )

            scores = self.P[batch_users] @ self.Q.T

            scores[rows, cols] = -np.inf

            idx = np.argpartition(-scores, top_n - 1, axis=1)[:, :top_n]
            top_items = idx[
                np.arange(len(idx))[:, None],
                np.argsort(-scores[np.arange(len(idx))[:, None], idx], axis=1)
            ]

            predictions[batch_users] = top_items

        return predictions

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