from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
from sklearn.metrics.pairwise import cosine_similarity

from src.base import BaseModel

# Jaccard similarity for binary feedback matrices
def jaccard_similarity(matrix: csr_matrix, batch_size: int = 1024) -> csr_matrix:
    matrix_bin = matrix.astype("bool").astype("int8")
    row_sums = np.asarray(matrix_bin.sum(axis=1)).ravel()
    n_rows = matrix_bin.shape[0]

    blocks: list[csr_matrix] = []
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        block = matrix_bin[start:stop]
        inter = block @ matrix_bin.T

        if inter.nnz:
            inter = inter.tocsr().astype(np.float32)
            data = inter.data
            indices = inter.indices
            indptr = inter.indptr
            for i in range(stop - start):
                row_start, row_stop = indptr[i], indptr[i + 1]
                if row_start == row_stop:
                    continue
                denom = row_sums[start + i] + row_sums[indices[row_start:row_stop]] - data[row_start:row_stop]
                data[row_start:row_stop] = np.divide(
                    data[row_start:row_stop],
                    denom,
                    out=np.zeros_like(data[row_start:row_stop], dtype=float),
                    where=denom != 0,
                )
        blocks.append(inter)

    if not blocks:
        return csr_matrix(matrix_bin.shape, dtype=float)
    return vstack(blocks).tocsr()

# Similarity with zeroed diagonal
def similarity_zd(matrix: csr_matrix, similarity_type: str) -> csr_matrix:
    if similarity_type == "cosine":
        similarity = cosine_similarity(matrix, dense_output=False)
    elif similarity_type == "jaccard":
        similarity = jaccard_similarity(matrix)
    else:
        raise ValueError(
            f"Unknown similarity type: {similarity_type}. "
            "Available types: 'cosine', 'jaccard'."
        )
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
    def __init__(
        self,
        n_neighbors: int | None = None,
        similarity_type: str = "cosine",
        downvote_seen_items: bool = True,
        name: str = "item_knn",
        **kwargs,
    ):
        super().__init__(name=name)
        self.n_neighbors = n_neighbors
        self.similarity_type = similarity_type
        self.downvote_seen_items = downvote_seen_items
        self.item_similarity: csr_matrix | None = None
        self._train_matrix: csr_matrix | None = None
        self._eval_users: np.ndarray | None = None

    def fit(self, train_dataset: Any, val_dataset: Any | None = None) -> "ItemKNNModel":
        coo = train_dataset.get_coo_array()
        n_users = getattr(train_dataset, "n_users", None)
        self._train_matrix = coo.tocsr() if hasattr(coo, "tocsr") else csr_matrix(coo)
        self._eval_users = np.arange(n_users if n_users is not None else self._train_matrix.shape[0], dtype=np.int64)
        item_similarity = similarity_zd(self._train_matrix.T, self.similarity_type)
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

            if self.downvote_seen_items and mask.any():
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
        self.similarity_type = trial.suggest_categorical("similarity_type", ["cosine", "jaccard"])
        self.downvote_seen_items = trial.suggest_categorical("downvote_seen_items", [True, False])
        return {
            "n_neighbors": self.n_neighbors,
            "similarity_type": self.similarity_type,
            "downvote_seen_items": self.downvote_seen_items,
        }

    def _score(self, user_item_matrix: csr_matrix) -> np.ndarray:
        scores = user_item_matrix.dot(self.item_similarity)
        return scores.toarray()
