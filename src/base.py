import optuna
import numpy as np

from abc import abstractmethod


class BaseDataset:
    """
    Base class for all datasets.
    """
    pass


class BaseModel:
    """
    Base class for all models.
    """
    def __init__(self, name: str, *args, **kwargs):
        self.name = name

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    @abstractmethod
    def fit(self, dataset: BaseDataset):
        """
        Fit the model to the dataset.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def predict(self, dataset: BaseDataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """
        Save the model checkpoint to the specified path.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        """
        Load the model checkpoint from the specified path.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def sample_params(self, trial: optuna.trial.Trial):
        """
        Sample hyperparameters for the model using the given trial.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseMetric:
    """
    Base class for all metrics.
    """
    def __init__(self, name: str, *args, **kwargs):
        self.name = name

    @abstractmethod
    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the metric given predictions and targets.

        Inputs:
            predictions (np.ndarray): The model predictions.
            targets (np.ndarray): The ground truth targets
        Returns:
            The computed metric value.
        """
        raise NotImplementedError("Subclasses should implement this method.")
