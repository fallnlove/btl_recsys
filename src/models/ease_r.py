import numpy as np
from scipy import sparse
import optuna
import os
import json

from src.base import BaseModel

class EASE_R(BaseModel):
    """
    EASE_R class.
    """
    regularization: float
    W: np.ndarray

    def __init__(self, regularization, name: str = "EASE R"):
        super().__init__(name)
        self.regularization = regularization

    def fit(self, train_dataset, val_dataset):
        """
        Fit the model to the dataset.
        """
        n_items = train_dataset.n_items

        A = train_dataset.get_coo_array().astype(np.float32)

        P_inv = (A.T.dot(A) + self.regularization * sparse.identity(n_items, format='csr')).toarray()
        try:
            P = np.linalg.inv(P_inv)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(P_inv)
        P_diag = np.diag(1 / P.diagonal())

        self.W = np.eye(n_items) - P.dot(P_diag)

    def predict(self, dataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.
        """
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

            batch_scores = batch_interactions.dot(self.W)

            mask = batch_interactions > 0
            batch_scores[mask] = -np.inf

            idx = np.argpartition(-batch_scores, top_n - 1, axis=1)[:, :top_n]
            top_indices_batch = idx[np.arange(len(idx))[:, None],
            np.argsort(-batch_scores[np.arange(len(idx))[:, None], idx], axis=1)]

            predictions[batch_users] = top_indices_batch

        return predictions

    def save_checkpoint(self, path: str):
        """
        Save the model checkpoint to the specified path.
        """
        os.makedirs(path, exist_ok=True)

        meta = {
            "W": "W.npy"
        }

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

        np.save(os.path.join(path, "W.npy"), self.W)

    def load_checkpoint(self, path: str):
        """
        Load the model checkpoint from the specified path.
        """
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        self.W = np.load(os.path.join(path, meta["W"]))

    def sample_params(self, trial: optuna.trial.Trial):
        """
        Sample hyperparameters for the model using the given trial.
        """
        params = {
            'regularization': trial.suggest_float(
                name='regularization',
                low=1e-5,
                high=1.0,
                log=True
            )
        }
        return params