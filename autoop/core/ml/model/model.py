
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(Artifact, ABC):
    """Base class for all models, both regression and classification."""

    def __init__(self, model_type: Literal["regression", "classification"], **kwargs):
        super().__init__(type="model", **kwargs)
        self.model_type = model_type
        self.parameters = None  # Placeholder for model parameters

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the model."""
        pass

    def save(self) -> None:
        """Save the model parameters (serialization logic can be added here)."""
        self.parameters = deepcopy(self)
        
    def load(self) -> None:
        """Load the model parameters (deserialization logic can be added here)."""
        if self.parameters is None:
            raise ValueError("Model has not been saved.")
        self.__dict__.update(deepcopy(self.parameters.__dict__))