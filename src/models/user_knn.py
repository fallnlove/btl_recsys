from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable, Any

import numpy as np
import optuna
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

from src.base import BaseModel

# Cosine similarity with zeroed diagonal
def cosine_similarity_zd(matrix: csr_matrix) -> csr_matrix:
    similarity = cosine_similarity(matrix, dense_output=False)
    similarity.setdiag(0)
    similarity.eliminate_zeros()
    return similarity

# Keep at most k neighbours per row for a sparse similarity matrix
def truncate_similarity(similarity: csr_matrix, k: int) -> csr_matrix:
    similarity = similarity.tocsr()
    inds = similarity.indices
    ptrs = similarity.indptr
    data = similarity.data
    new_ptrs = [0]
    new_inds: list[int] = []
    new_data: list[np.ndarray] = []
    for i in range(len(ptrs) - 1):
        start, stop = ptrs[i], ptrs[i + 1]
        if start < stop:
            data_slice = data[start:stop]
            topk = min(len(data_slice), k)
            # Finding the top-k indices
            idx = np.argpartition(data_slice, -topk)[-topk:]
            new_data.append(data_slice[idx])
            new_inds.append(inds[idx + start])
            new_ptrs.append(new_ptrs[-1] + len(idx))
        else:
            new_ptrs.append(new_ptrs[-1])
    if not new_data:
        return csr_matrix(similarity.shape)
    new_data_arr = np.concatenate(new_data)
    new_inds_arr = np.concatenate(new_inds)
    return csr_matrix((new_data_arr, new_inds_arr, new_ptrs), shape=similarity.shape)

class UserKNNModel(BaseModel):
    def __init__(self, n_neighbors: int | None = None, name: str = "user_knn", **kwargs):
        super().__init__(name=name)
        self.n_neighbors = n_neighbors
        self.similarity: csr_matrix | None = None
        self._train_matrix: csr_matrix | None = None
        self._eval_users: np.ndarray | None = None

    def fit(self, train_dataset: Any, val_dataset: Any | None = None) -> "UserKNNModel":
        coo = train_dataset.get_coo_array()
        n_users = getattr(train_dataset, "n_users", None)
        self._train_matrix = coo.tocsr() if hasattr(coo, "tocsr") else csr_matrix(coo)
        self._eval_users = np.arange(n_users if n_users is not None else self._train_matrix.shape[0], dtype=np.int64)
        similarity = cosine_similarity_zd(self._train_matrix)
        if self.n_neighbors is not None:
            similarity = truncate_similarity(similarity, self.n_neighbors)
        self.similarity = similarity
        return self

    def predict(self, dataset: Any, top_n: int) -> np.ndarray:
        n_users = getattr(dataset, "n_users", self._train_matrix.shape[0])
        predictions = np.full((n_users, top_n), fill_value=-1, dtype=np.int64)

        dataloader_fn = getattr(dataset, "get_dataloader", None)
        loader = dataloader_fn(batch_size=256, shuffle=False)
        for batch in loader:
            batch_user_ids = batch["user_id"].numpy()
            # Compute scores for all candidates for this batch of users
            scores_dense = self._score(batch_user_ids).astype(float)

            history = batch["history"].numpy()
            mask = history != -1
            if mask.any():
                row_idx = np.repeat(np.arange(history.shape[0]), mask.sum(axis=1))
                col_idx = history[mask]
                # Mask already seen items so they never enter top-N
                scores_dense[row_idx, col_idx] = -np.inf

            current_top_n = min(top_n, scores_dense.shape[1])
            top_idx = np.argpartition(-scores_dense, kth=current_top_n - 1, axis=1)[:, :current_top_n]
            row_idx = np.arange(top_idx.shape[0])[:, None]
            top_sorted = top_idx[row_idx, np.argsort(-scores_dense[row_idx, top_idx], axis=1)]
            predictions[batch_user_ids, :current_top_n] = top_sorted

        return predictions

    def save_checkpoint(self, path: str):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(save_path, self.similarity)

    def load_checkpoint(self, path: str):
        self.similarity = load_npz(path)

    def sample_params(self, trial: optuna.trial.Trial):
        self.n_neighbors = trial.suggest_int("n_neighbors", 5, 200, step=5)
        return {"n_neighbors": self.n_neighbors}

    def _score(self, user_ids: np.ndarray) -> np.ndarray:
        user_similarity = self.similarity[user_ids]
        scores = user_similarity.dot(self._train_matrix)
        return scores.toarray()