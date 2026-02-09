from src.base import BaseModel
import numpy as np
import json
import os
import optuna
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import scipy.sparse
import torch
from src.models.utils.sparse_svd_gpu import gpu_sparse_svd
from src.metrics import NDCGMetric

import warnings

def rank_schedule(rank_max):
    yield from [1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10]

    k = 10
    while k < rank_max:
        k += k // 10
        yield min(k, rank_max)

def _rescale_matrix(matrix, s):
    nnz_per_col = matrix.count_nonzero(axis=0)
    D = sp.diags(nnz_per_col ** (0.5 * (s - 1)))
    return matrix @ D

def _gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

class PureSVDModel(BaseModel):

    def __init__(
        self,
        name: str = "pure_svd_model",
        rank_max: int = 1500,
        s: float = 0.0,
        val_top_n: int = 20,
    ):
        super().__init__(name)
        self.rank_max = rank_max
        self.rank = None
        self.s = s
        self.val_top_n = val_top_n
        self.proj = None
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def fit(self, train_dataset, val_dataset):
        matrix = train_dataset.get_coo_array()
        matrix = _rescale_matrix(matrix, self.s)
        self.rank_max = min(self.rank_max, min(matrix.shape) - 20)

        if _gpu_available():
            print("[SVD] Using GPU (CuPy)")
            _, _, Vt = gpu_sparse_svd(
                matrix,
                k=self.rank_max,
                return_numpy=True,
            )
        else:
            print("[SVD] Using CPU (SciPy)")
            _, _, Vt = svds(matrix, k=self.rank_max)

        V_full = Vt.T  # (n_items, rank_max)
        
        if val_dataset is None:
            print("[PureSVD] No validation dataset provided, using max rank =",self.rank_max)
            self.rank = self.rank_max
            self.proj = V_full
            return self

        metric = NDCGMetric(self.val_top_n)
        holdout_users = val_dataset.get_holdout_users()
        y_true = val_dataset.get_holdout_array()
        assert holdout_users.max() < val_dataset.n_users
        assert val_dataset.n_items == train_dataset.n_items
        assert val_dataset.n_items == V_full.shape[0]

        best_score = -np.inf
        best_rank = 1

        for k in rank_schedule(self.rank_max):
            self.proj = V_full[:, :k]
            preds = self.predict(val_dataset, self.val_top_n)
            score = metric(preds[holdout_users], y_true[holdout_users])

            if score > best_score:
                best_score = score
                best_rank = k

        self.rank = best_rank
        self.proj = V_full[:, :best_rank]

        print(f"[PureSVD] Best rank = {best_rank}, NDCG = {best_score:.6f}")
        return self

        

    def predict(self, dataset, top_n: int) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        V = torch.from_numpy(self.proj).to(device)
        Vt = V.T

        num_users = dataset.n_users
        num_items = dataset.n_items

        data_loader = dataset.get_dataloader(batch_size=1024, shuffle=False)

        top_indices = np.zeros((num_users, top_n), dtype=np.int32)

        for batch in data_loader:
            user_ids = batch["user_id"]
            history = batch["history"]

            B = len(user_ids)

            hist = torch.as_tensor(history, device=device)
            mask = hist != -1

            rows = torch.arange(B, device=device).unsqueeze(1).expand_as(hist)
            rows = rows[mask]
            cols = hist[mask]

            R = torch.zeros((B, num_items), device=device)
            R[rows, cols] = 1.0

            scores = (R @ V) @ Vt
            scores[rows, cols] = -torch.inf

            topk = torch.topk(scores, k=top_n, dim=1).indices
            top_indices[user_ids.cpu().numpy()] = topk.cpu().numpy()

        return top_indices



    def save_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)

        meta = {
            "rank": self.rank,
            "proj": "proj.npy"
        }

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

        np.save(os.path.join(path, "proj.npy"), self.proj)

    def load_checkpoint(self, path: str):
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        self.rank = int(meta["rank"])
        self.proj = np.load(os.path.join(path, meta["proj"]))
        
    def suggest_additional_params(self) -> dict:
        if self.rank is None:
            return {}
        return {"rank_max": int(self.rank)}
