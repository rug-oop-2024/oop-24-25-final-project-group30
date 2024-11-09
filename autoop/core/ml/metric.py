from abc import ABC, abstractmethod
from typing import Any
import numpy as np

# List of available metrics for reference
METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]

def get_metric(name: str):
    """Factory function to get a metric by name.
    
    Args:
        name (str): The name of the metric to retrieve.
    
    Returns:
        Metric: An instance of the requested metric.
        
    Raises:
        ValueError: If the metric name is not recognized.
    """
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r2_score":
        return R2Score()
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "f1_score":
        return F1Score()
    else:
        raise ValueError(f"Unknown metric name: {name}")

class Metric(ABC):
    """Base class for all metrics."""
    
    @abstractmethod
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute the metric.
        
        Args:
            predictions (np.ndarray): The model predictions.
            ground_truth (np.ndarray): The ground truth values.
        
        Returns:
            float: The computed metric value.
        """
        pass

# Regression Metrics
class MeanSquaredError(Metric):
    """Mean Squared Error (MSE) metric."""
    
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.mean((predictions - ground_truth) ** 2)

class MeanAbsoluteError(Metric):
    """Mean Absolute Error (MAE) metric."""
    
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.mean(np.abs(predictions - ground_truth))

class R2Score(Metric):
    """R² Score (Coefficient of Determination) metric."""
    
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        ss_total = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        ss_residual = np.sum((ground_truth - predictions) ** 2)
        return 1 - (ss_residual / ss_total)

# Classification Metrics
class Accuracy(Metric):
    """Accuracy metric for classification."""
    
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.mean(predictions == ground_truth)

class Precision(Metric):
    """Precision metric for binary classification."""
    
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        true_positives = np.sum((predictions == 1) & (ground_truth == 1))
        predicted_positives = np.sum(predictions == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0

class Recall(Metric):
    """Recall metric for binary classification."""
    
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        true_positives = np.sum((predictions == 1) & (ground_truth == 1))
        actual_positives = np.sum(ground_truth == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0.0

class F1Score(Metric):
    """F1 Score metric for binary classification."""
    
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        precision = Precision()(predictions, ground_truth)
        recall = Recall()(predictions, ground_truth)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    