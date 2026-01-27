import numpy as np
import optuna
from implicit import als
from scipy import sparse

from src.base import BaseModel
from src.datasets import RecSysDataset


class ALS_Implicit(BaseModel):
    """
    ALS FM model
    http://yifanhu.net/PUB/cf.pdf
    """

    def __init__(self, name: str, *args, **kwargs):
        super(ALS_Implicit, self).__init__(name=name)
        self.name = name
        self.alpha = kwargs.get('alpha', 1)
        self.lambda_ = kwargs.get('lambda_', 1)
        self.rank = kwargs.get('rank', 1)
        self.n_iters = kwargs.get('n_iters', 1)
        self.random_seed = kwargs.get('random_seed', 0)
        self.num_threads = kwargs.get('num_threads', 0)
        self.model = als.AlternatingLeastSquares(factors=self.rank, regularization=self.lambda_,
                                                 alpha=self.alpha, dtype=np.float32, use_native=True,
                                                 use_cg=True, iterations=self.n_iters,
                                                 calculate_training_loss=False, num_threads=self.num_threads,
                                                 random_state=self.random_seed)
        #self.pred_batchsize = 2

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def fit(self, dataset: RecSysDataset, val):
        """
        Fit the model to the dataset.
        """
        np.random.seed(self.random_seed)
        user_item_data = sparse.csr_matrix(dataset.get_coo_array_rating().astype(np.float32).tocsr())
        self.model.fit(user_item_data, show_progress=False)

    def predict(self, dataset: RecSysDataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.
        """
        recs = np.repeat(np.arange(top_n, dtype=int), repeats=dataset.n_users)
        recs = recs.reshape(top_n, -1).T
        userid = np.unique(dataset.get_holdout_users())
        user_item_data = dataset.get_coo_array_rating().astype(np.float32).tocsr()
        user_item_data = sparse.csr_matrix(user_item_data)
        rec, _ = self.model.recommend(userid, user_item_data[userid], N=top_n,
                                      filter_already_liked_items=True)
        recs[userid] = rec
        '''
        for batch in np.array_split(userid, len(userid)//self.pred_batchsize):
            data = sparse.csr_matrix(user_item_data[batch])
            rec, _ = self.model.recommend(batch, data, N=top_n,
                                                  filter_already_liked_items=True)
            recs[batch] = rec
        '''
        return recs

    def save_checkpoint(self, path: str):
        """
        Save the model checkpoint to the specified path.
        """
        pass

    def load_checkpoint(self, path: str):
        """
        Load the model checkpoint from the specified path.
        """
        pass

    def sample_params(self, trial: optuna.trial.Trial):
        """
        Sample hyperparameters for the model using the given trial.
        """
        raise NotImplementedError("Subclasses should implement this method.")
