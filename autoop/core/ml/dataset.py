from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io

class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, version: str="1.0.0"):
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
        
    def read(self) -> pd.DataFrame:
        """Reads the dataset data as a DataFrame."""
        bytes = super().read()
        try:
            csv = bytes.decode('utf-8')
        except UnicodeDecodeError:
            csv = bytes.decode('latin1')
        
        # Use on_bad_lines='skip' to handle lines with incorrect fields
        return pd.read_csv(io.StringIO(csv), on_bad_lines='skip')
    
    def save(self, data: pd.DataFrame) -> bytes:
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
    