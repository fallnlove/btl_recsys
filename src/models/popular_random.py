import optuna
import numpy as np
from torch.utils.data import DataLoader

from src.base import BaseModel
from src.utils.collate import collate_fn

class PopularRandom(BaseModel):
    """
    Base class for all models.
    """
    def __init__(self, name: str = "PopularRandom"):
        super(PopularRandom, self).__init__(name)

    def fit(self, train_dataset, val_dataset):
        """
        Fit the model to the dataset.
        """

        n_items = train_dataset.n_items
        self.item_counts = np.zeros(n_items, dtype=np.long)
        for user_id, item_id in zip(*train_dataset.get_coo_array().coords):
            self.item_counts[item_id] += 1

    def predict(self, dataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.
        """
        predictions = np.zeros((dataset.n_users, top_n), dtype=np.long)
        dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
        for batch in dataloader:
            probs = np.copy(self.item_counts).astype(np.float32)
            probs[batch['history'][0].numpy().astype(np.long).tolist()] = 0
            predictions[batch['user_id'].numpy()] = np.random.choice(
                dataset.n_items,
                size=(top_n, ),
                replace=False,
                p=probs / probs.sum(),
            )
        return predictions

    def save_checkpoint(self, path: str):
        """
        Save the model checkpoint to the specified path.
        """
        np.save(path, self.item_counts)

    def load_checkpoint(self, path: str):
        """
        Load the model checkpoint from the specified path.
        """
        self.item_counts = np.load(path)

    @staticmethod
    def sample_params(trial: optuna.trial.Trial):
        """
        Sample hyperparameters for the model using the given trial.
        """
        pass
