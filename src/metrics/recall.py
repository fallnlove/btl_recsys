import numpy as np

from src import BaseMetric
from src.utils.metric_utils import check_unique


class RecallMetric(BaseMetric):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__(name=f"recall@{k}", *args, **kwargs)
        self._k = k

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        assert check_unique(predictions), "Predicted items must be unique per user."
        preds = predictions[:, :self._k]
        hits = (preds == targets[:, None]).astype(float)
        recall = np.sum(hits, axis=-1)

        return float(np.mean(recall))
