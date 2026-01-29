import torch
import numpy as np
from tqdm import tqdm

from src.base import BaseModel
from src.models.utils.repeatable_rec import is_repeatable

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

        self.item_counts = train_dataset.get_coo_array().sum(axis=0)

    def predict(self, dataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_probs = torch.as_tensor(self.item_counts, dtype=torch.float32, device=device)
        n_users = dataset.n_users
        n_items = dataset.n_items

        predictions = np.zeros((n_users, top_n), dtype=np.int64)

        dataloader = dataset.get_dataloader(batch_size=32, shuffle=False)

        for batch in tqdm(dataloader, desc="Predicting"):
            user_ids = batch["user_id"].to(device)            # shape: (B,)
            history = batch["history"].to(device)             # shape: (B, L)

            B = user_ids.size(0)

            probs = base_probs.expand(B, n_items).clone()     # shape: (B, n_items)

            if not is_repeatable(dataset):
                mask = history >= 0
                padded_history = history.clone()
                padded_history[~mask] = 0
                probs.scatter_(1, padded_history, 0.0)
            sampled = torch.multinomial(
                probs,
                num_samples=top_n,
                replacement=False
            )   # shape: (B, top_n)
            predictions[user_ids.cpu().numpy()] = sampled.cpu().numpy()

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
