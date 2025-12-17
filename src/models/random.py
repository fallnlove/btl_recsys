import torch
import numpy as np
from tqdm import tqdm

from src.base import BaseModel


class RandomModel(BaseModel):

    def __init__(self, name: str = "RandomModel"):
        super().__init__(name)

    def fit(self, train_dataset, val_dataset):
        pass

    def predict(self, dataset, top_n: int) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_users = dataset.n_users
        n_items = dataset.n_items

        predictions = np.zeros((n_users, top_n), dtype=np.int64)

        base_probs = torch.ones(n_items, device=device)

        dataloader = dataset.get_dataloader(batch_size=32, shuffle=False)

        for batch in tqdm(dataloader, desc="Predicting"):
            user_ids = batch["user_id"].to(device)    # (B,)
            history = batch["history"].to(device)     # (B, L)

            B = user_ids.size(0)

            probs = base_probs.expand(B, n_items).clone()

            mask = history >= 0
            padded_history = history.clone()
            padded_history[~mask] = 0
            probs.scatter_(1, padded_history, 0.0)

            sampled = torch.multinomial(
                probs,
                num_samples=top_n,
                replacement=False
            )

            predictions[user_ids.cpu().numpy()] = sampled.cpu().numpy()

        return predictions

    def save_checkpoint(self, path: str):
        """
        Nothing to save for random model.
        """
        pass

    def load_checkpoint(self, path: str):
        """
        Nothing to load for random model.
        """
        pass
