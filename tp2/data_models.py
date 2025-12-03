from pydantic import BaseModel
from typing import List
import numpy as np


class RagMetadata(BaseModel):
    text: str
    source: str


class EmbeddingDatabaseEntry(BaseModel):
    id: str
    values: np.ndarray
    metadata: RagMetadata

    class Config:
        arbitrary_types_allowed = True
