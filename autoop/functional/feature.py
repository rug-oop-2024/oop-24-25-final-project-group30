
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd
import io

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Detects feature types (numerical or categorical) for a given dataset.
    
    Args:
        dataset: Dataset instance to detect features from.
    
    Returns:
        List[Feature]: Detected features with types.
    """

    df = dataset.read()

    features = []
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            
            feature_type = 'numerical'
        else:
            feature_type = 'categorical'
        features.append(Feature(name=column, type=feature_type))
    
    return features
