from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
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

class ItemKNNModel(BaseModel):
    def __init__(self, n_neighbors: int | None = None, name: str = "item_knn", **kwargs):
        super().__init__(name=name)
        self.n_neighbors = n_neighbors
        self.item_similarity: csr_matrix | None = None
        self._train_matrix: csr_matrix | None = None
        self._eval_users: np.ndarray | None = None

    def fit(self, train_dataset: Any, val_dataset: Any | None = None) -> "ItemKNNModel":
        coo = train_dataset.get_coo_array()
        n_users = getattr(train_dataset, "n_users", None)
        self._train_matrix = coo.tocsr() if hasattr(coo, "tocsr") else csr_matrix(coo)
        self._eval_users = np.arange(n_users if n_users is not None else self._train_matrix.shape[0], dtype=np.int64)
        item_similarity = cosine_similarity_zd(self._train_matrix.T)
        if self.n_neighbors is not None:
            item_similarity = truncate_similarity(item_similarity, self.n_neighbors)
        self.item_similarity = item_similarity
        return self

    def predict(self, dataset: Any, top_n: int) -> np.ndarray:
        n_users = getattr(dataset, "n_users", self._train_matrix.shape[0])
        n_items = self._train_matrix.shape[1]
        predictions = np.full((n_users, top_n), fill_value=-1, dtype=np.int64)

        dataloader_fn = getattr(dataset, "get_dataloader", None)
        loader = dataloader_fn(batch_size=256, shuffle=False)
        for batch in loader:
            batch_user_ids = batch["user_id"].numpy()
            history = batch["history"].numpy()
            mask = history != -1

            if mask.any():
                # Build CSR history matrix for the batch
                rows = np.repeat(np.arange(history.shape[0]), mask.sum(axis=1))
                cols = history[mask]
                data = np.ones_like(cols, dtype=np.float32)
                batch_user_item = csr_matrix((data, (rows, cols)), shape=(history.shape[0], n_items))
            else:
                batch_user_item = csr_matrix((history.shape[0], n_items))

            # Compute candidate scores for this batch
            scores_dense = self._score(batch_user_item).astype(float)

            if mask.any():
                rows_mask = np.repeat(np.arange(history.shape[0]), mask.sum(axis=1))
                cols_mask = history[mask]
                # Mask seen items so they never enter top-N
                scores_dense[rows_mask, cols_mask] = -np.inf

            current_top_n = min(top_n, scores_dense.shape[1])
            top_idx = np.argpartition(-scores_dense, kth=current_top_n - 1, axis=1)[:, :current_top_n]
            row_idx = np.arange(top_idx.shape[0])[:, None]
            top_sorted = top_idx[row_idx, np.argsort(-scores_dense[row_idx, top_idx], axis=1)]
            predictions[batch_user_ids, :current_top_n] = top_sorted

        return predictions

    def save_checkpoint(self, path: str):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(save_path, self.item_similarity)

    def load_checkpoint(self, path: str):
        self.item_similarity = load_npz(path)

    def sample_params(self, trial: optuna.trial.Trial):
        self.n_neighbors = trial.suggest_int("n_neighbors", 5, 200, step=5)
        return {"n_neighbors": self.n_neighbors}

    def _score(self, user_item_matrix: csr_matrix) -> np.ndarray:
        scores = user_item_matrix.dot(self.item_similarity)
        return scores.toarray()