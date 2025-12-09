import numpy as np

from src import BaseMetric
from src.utils.metric_utils import check_unique


class CoverageMetric(BaseMetric):
    def __init__(self, k: int, n_items: int, *args, **kwargs):
        super().__init__(name=f"coverage@{k}", *args, **kwargs)
        self._k = k
        self._n_items = n_items

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        assert check_unique(predictions), "Predicted items must be unique per user."
        preds = predictions[:, :self._k]
        unique_items = np.unique(preds)
        coverage = len(unique_items) / self._n_items

        return float(coverage)
