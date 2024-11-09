
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    """
    A class used to represent a Feature.

    Attributes:
    ----------
    name : str
        The name of the feature.
    type : Literal["numerical", "categorical"]
        The type of the feature, which can be either "numerical" or "categorical".

    Methods:
    -------
    __str__():
        Returns a string representation of the feature, including its name and type.
    """
    name: str
    type: Literal["numerical", "categorical"]

    def __init__(self, name: str, type: Literal["numerical", "categorical"]):
        super().__init__(name=name, type=type)

    def __str__(self) -> str:
        """Returns a string representation of the feature, including its name and type."""
        return f"Feature(name='{self.name}', type='{self.type}')"