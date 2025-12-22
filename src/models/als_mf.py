import numpy as np
import optuna
# import time

from src.base import BaseModel
from src.datasets import RecSysDataset


def run_als_sparse(Cm1r, CPr, Cm1c, CPc, X, Y, n_iters, lambda_, users, batchsize=1024):
    # print('shape', X.shape, Y.shape)
    # start_time = time.time()
    I = np.eye(Y.shape[1], dtype=np.float32) * lambda_
    absent_users = set(range(X.shape[0]))
    absent_users.difference_update(set(users))
    absent_users = list(absent_users)
    user_batches = np.array_split(users, len(users) // batchsize)
    items = np.arange(Y.shape[0]).astype(int)
    item_batches = np.array_split(items, len(items) // batchsize)
    for iteration in range(n_iters):
        YtY = Y.T @ Y + I
        cnt = 0
        for batch in user_batches:
            y = []
            for user in batch:
                y.append(Cm1r[[user], :].multiply(Y.T).dot(Y))
            y = np.stack(y)
            M = y + YtY[None, :, :]
            a = CPr[batch, :].dot(Y)
            X[batch, :] = np.linalg.solve(M, a[:, :, None]).squeeze(-1)
            cnt += 1
        X[absent_users] = 0
        XtX = X.T @ X + I
        cnt = 0
        for batch in item_batches:
            x = []
            for item in batch:
                x.append(Cm1c[:, [item]].T.multiply(X.T).dot(X))
            x = np.stack(x)
            M = x + XtX[None, :, :]
            a = CPc[:, batch].T.dot(X)
            Y[batch, :] = np.linalg.solve(M, a[:, :, None]).squeeze(-1)
            cnt += 1
        # print('Time', iteration, time.time()-start_time)
    return X, Y


class ALSMF_sparse(BaseModel):
    """
    ALS FM model
    http://yifanhu.net/PUB/cf.pdf
    """

    def __init__(self, name: str, *args, **kwargs):
        super(ALSMF_sparse, self).__init__(name=name)
        self.name = name
        self.alpha = kwargs.get('alpha', 1)
        self.lambda_ = kwargs.get('lambda_', 1)
        self.rank = kwargs.get('rank', 1)
        self.n_iters = kwargs.get('n_iters', 1)
        self.batchsize = kwargs.get('batchsize', 2048)
        self.random_seed = kwargs.get('random_seed', 0)
        self.pred_batchsize = 2

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def fit(self, dataset: RecSysDataset, val):
        """
        Fit the model to the dataset.
        """
        np.random.seed(self.random_seed)
        self.X = np.zeros((dataset.n_users, self.rank)).astype(np.float32)
        self.Y = np.random.randn(dataset.n_items, self.rank).astype(np.float32)
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        Cm1r = self.alpha * dataset.get_coo_array_rating().astype(np.float32).tocsr()
        CPr = Cm1r + dataset.get_coo_array().astype(np.float32).tocsr()
        Cm1c = self.alpha * dataset.get_coo_array_rating().astype(np.float32).tocsc()
        CPc = Cm1c + dataset.get_coo_array().astype(np.float32).tocsc()
        self.X, self.Y = run_als_sparse(Cm1r, CPr, Cm1c, CPc, self.X, self.Y, self.n_iters, self.lambda_,
                                        dataset._users, self.batchsize)
        self.YtY = self.Y.T @ self.Y + self.lambda_ * np.eye(self.rank, dtype=np.float32)

    def predict(self, dataset: RecSysDataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.
        """
        n_users, n_items = dataset.n_users, dataset.n_items
        Cm1 = self.alpha * dataset.get_coo_array_rating().astype(np.float32).tocsr()
        CP = Cm1 + dataset.get_coo_array().astype(np.float32).tocsr()
        recs = np.repeat(np.arange(top_n, dtype=int), repeats=n_users)
        recs = recs.reshape(top_n, -1).T
        users = np.unique(dataset.get_holdout_users())
        batches = np.array_split(users, max(len(users) // self.pred_batchsize, 1))
        for batch in batches:
            y = []
            for user in batch:
                y.append(Cm1[[user], :].multiply(self.Y.T).dot(self.Y))
            M = np.stack(y) + self.YtY[None, :, :]
            a = CP[batch, :].dot(self.Y)
            score = (self.Y @ np.linalg.solve(M, a[:, :, None])).squeeze(-1)
            row_indices, col_indices = Cm1[batch, :].nonzero()
            score[row_indices, col_indices] = -np.inf
            idx = np.argpartition(score, -top_n, axis=-1)[:, -top_n:]
            recs[batch] = idx[np.arange(len(idx))[:, None],
                              np.argsort(-score[np.arange(len(idx))[:, None], idx], axis=1)]
        return recs

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
