import numpy as np

from src import BaseMetric


class HitRateMetric(BaseMetric):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__(name=f"hit_rate@{k}", *args, **kwargs)
        self._k = k

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError("HitRateMetric is not implemented yet.")
