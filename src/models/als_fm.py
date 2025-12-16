import numpy as np
import scipy
import optuna
from numba import njit
from torch.utils.data import DataLoader

from src.base import BaseModel
from src.datasets import RecSysDataset
from src.utils.collate import collate_fn


def run_als_sparse(Cm1, CP, X, Y, n_iters, lambda_):
    Iy, Ix = np.eye(Y.shape[1], dtype=np.float32), np.eye(X.shape[1], dtype=np.float32)
    for iteration in range(n_iters):
        YtY = Y.T @ Y
        for u in range(X.shape[0]):
            M = (YtY + (Cm1[[u], :].multiply(Y.T)) @ Y) + lambda_ * Iy
            a = CP[[u], :].dot(Y).T
            X[u, :] = np.linalg.solve(M, a[:, 0])
        XtX = X.T @ X
        for i in range(Y.shape[0]):
            N = (XtX + (Cm1[:, [i]].T.multiply(X.T)) @ X) + lambda_ * Ix
            a = CP[:, [i]].T.dot(X)
            Y[i, :] = np.linalg.solve(N, a[0, :])
    return X, Y


class ALSFM_sparse(BaseModel):
    """
    ALS FM model
    http://yifanhu.net/PUB/cf.pdf
    """

    def __init__(self, name: str, *args, **kwargs):
        super(ALSFM_sparse, self).__init__(name=name)
        self.name = name
        self.alpha = kwargs.get('alpha', 1)
        self.lambda_ = kwargs.get('lambda_', 1)
        self.rank = kwargs.get('rank', 1)
        self.n_iters = kwargs.get('n_iters', 1)
        self.random_seed = kwargs.get('random_seed', 0)

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def fit(self, dataset: RecSysDataset, val):
        """
        Fit the model to the dataset.
        """
        np.random.seed(self.random_seed)
        self.X = np.random.randn(dataset.n_users, self.rank).astype(np.float32)
        self.Y = np.random.randn(dataset.n_items, self.rank).astype(np.float32)
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        Cm1 = self.alpha * dataset.get_coo_array_rating()
        CP = Cm1 + dataset.get_coo_array()
        self.X, self.Y = run_als_sparse(Cm1, CP, self.X, self.Y, self.n_iters, self.lambda_)

    def predict(self, dataset: RecSysDataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.
        """
        if dataset is None:
            return self.X[dataset._users, :].T @ self.Y
        Cm1 = self.alpha * dataset.get_coo_array_rating()
        CP = Cm1 + dataset.get_coo_array()
        Iy = np.eye(self.rank, dtype=np.float32)
        recs = []
        YtY = self.Y.T @ self.Y
        for user in range(self.n_users):
            if user in dataset._users:
                M = (YtY + (Cm1[[user], :].multiply(self.Y.T)) @ self.Y) + self.lambda_ * Iy
                a = CP[[user], :].dot(self.Y).T
                # X[:, u] = np.linalg.solve(M.astype(np.float32), a[:, 0])
                score = self.Y @ np.linalg.solve(M, a[:, 0])  # @ CP[user]
                recs.append(np.argsort(score)[-top_n:])
            else:
                recs.append(np.zeros(top_n))
        return np.array(recs)

    def save_checkpoint(self, path: str):
        """
        Save the model checkpoint to the specified path.
        """
        np.save(path, [self.X, self.Y])

    def load_checkpoint(self, path: str):
        """
        Load the model checkpoint from the specified path.
        """
        self.X, self.Y = np.load(path)

    def sample_params(self, trial: optuna.trial.Trial):
        """
        Sample hyperparameters for the model using the given trial.
        """
        raise NotImplementedError("Subclasses should implement this method.")



