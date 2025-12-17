from src.base import BaseModel
import numpy as np
import json
import os
import optuna
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import scipy.sparse
from src.models.utils.sparse_svd_gpu import gpu_sparse_svd

import warnings

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

    def __init__(self, name: str = "pure_svd_model", rank: int = 50, s=0):
        super().__init__(name)
        self.rank = rank
        self.s = s
        self.proj = None
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def fit(self, train_dataset, val_dataset):
        matrix = train_dataset.get_coo_array()
        matrix = _rescale_matrix(matrix, self.s)
        
        if _gpu_available():
            print("[SVD] Using GPU (CuPy)")
            _, _, Vt = gpu_sparse_svd(
                matrix,
                k=self.rank,
                return_numpy=True,
            )
        else:
            print("[SVD] Using CPU (SciPy)")
            _, _, Vt = svds(matrix, k=self.rank)
            
        self.proj = Vt.T
        

    def predict(self, dataset, top_n: int) -> np.ndarray:
        num_items = dataset.n_items
        num_users = dataset.n_users

        data_loader = dataset.get_dataloader(batch_size=1024, shuffle=False)
        top_indices = np.zeros((num_users, top_n), dtype=np.int32)
        
        for batch in data_loader:
            batch_users = batch['user_id']
            interactions = batch['history']
            batch_interactions = np.zeros((len(batch_users), num_items))
            
            arr = np.array(interactions)

            mask = arr != -1

            rows = np.repeat(np.arange(arr.shape[0]), mask.sum(axis=1))
            cols = arr[mask].ravel()

            batch_interactions[rows, cols] = 1
                        
            batch_scores = (self.proj @ (self.proj.T @ batch_interactions.T)).T
            
            
            idx = np.argpartition(-batch_scores, top_n-1, axis=1)[:, :top_n]
            top_indices_batch = idx[np.arange(len(idx))[:,None],
            np.argsort(-batch_scores[np.arange(len(idx))[:,None], idx], axis=1)]
            
            top_indices[batch_users] = top_indices_batch
            
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
        
