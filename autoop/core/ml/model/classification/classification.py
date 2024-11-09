from autoop.core.ml.model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class LogisticRegressionModel(Model):
    def __init__(self):
        super().__init__(model_type="classification")
        self.model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class DecisionTreeClassification(Model):
    def __init__(self):
        super().__init__(model_type="classification")
        self.model = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class RandomForestClassification(Model):
    def __init__(self):
        super().__init__(model_type="classification")
        self.model = RandomForestClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
