import numpy as np

from src import BaseMetric


class NDCGMetric(BaseMetric):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__(name=f"ndcg@{k}", *args, **kwargs)
        self._k = k

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError("NDCGMetric is not implemented yet.")
