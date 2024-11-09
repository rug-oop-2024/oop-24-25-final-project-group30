from autoop.core.ml.model import Model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

class MultipleLinearRegression(Model):
    def __init__(self):
        super().__init__(model_type="regression")
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

# Regression Model: Decision Tree Regressor
class DecisionTreeRegression(Model):
    def __init__(self):
        super().__init__(model_type="regression")
        self.model = DecisionTreeRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

# Regression Model: Random Forest Regressor
class RandomForestRegression(Model):
    def __init__(self):
        super().__init__(model_type="regression")
        self.model = RandomForestRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)