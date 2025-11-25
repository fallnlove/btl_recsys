import numpy as np

from src import BaseMetric


class CoverageMetric(BaseMetric):
    def __init__(self, k: int, n_items: int, *args, **kwargs):
        super().__init__(name=f"coverage@{k}", *args, **kwargs)
        self._k = k
        self._n_items = n_items

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        preds = predictions[:, :self._k]
        unique_items = np.unique(preds)
        coverage = len(unique_items) / self._n_items

        return float(coverage)
