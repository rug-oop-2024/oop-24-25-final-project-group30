
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.regression import (
    MultipleLinearRegression,
    DecisionTreeRegression,
    RandomForestRegression,
)
from autoop.core.ml.model.classification.classification import (
    LogisticRegressionModel,
    DecisionTreeClassification,
    RandomForestClassification,
)

REGRESSION_MODELS = [
    "multiple_linear_regression",
    "decision_tree_regression",
    "random_forest_regression",
]

CLASSIFICATION_MODELS = [
    "logistic_regression",
    "decision_tree_classification",
    "random_forest_classification",
]

def get_model(model_name: str):
    """Factory function to get a model by name."""
    if model_name == "multiple_linear_regression":
        return MultipleLinearRegression()
    elif model_name == "decision_tree_regression":
        return DecisionTreeRegression()
    elif model_name == "random_forest_regression":
        return RandomForestRegression()
    elif model_name == "logistic_regression":
        return LogisticRegressionModel()
    elif model_name == "decision_tree_classification":
        return DecisionTreeClassification()
    elif model_name == "random_forest_classification":
        return RandomForestClassification()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
