from pydantic import BaseModel, Field
import base64
from typing import Optional, Dict

class Artifact(BaseModel):
    asset_path: Optional[str] = Field(None, description="Path to the stored artifact.")
    version: str = Field("1.0.0", description="Version of the artifact.")
    data: Optional[str] = Field(None, description="Base64-encoded artifact data.")
    metadata: Optional[Dict[str, str]] = Field(None, description="Metadata related to the artifact.")
    type: str = Field(..., description="Type of the artifact, e.g., model, dataset.")
    #model_type: Optional[str] = None  

    def read(self) -> bytes:
        if not self.data:
            raise ValueError("No data to read in the artifact.")
        
        try:
            # Decode base64 data
            return base64.b64decode(self.data + '===')
        except base64.binascii.Error:
            raise ValueError("Data is not correctly base64 encoded")

    def save(self, data: bytes) -> None:
        """Encodes and saves the provided data in base64 format."""
        self.data = base64.b64encode(data).decode('utf-8')
