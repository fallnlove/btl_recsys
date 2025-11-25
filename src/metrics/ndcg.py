import numpy as np

from src import BaseMetric


class NDCGMetric(BaseMetric):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__(name=f"ndcg@{k}", *args, **kwargs)
        self._k = k

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        preds = predictions[:, :self._k]
        hits = (preds == targets[:, None]).astype(float)
        discount_factor = 1 / np.log2(np.arange(1, self._k + 1) + 1)
        dcg = hits @ discount_factor

        return float(np.mean(dcg))
