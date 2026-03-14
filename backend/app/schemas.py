from pydantic import BaseModel, Field, conlist
from typing import List, Optional

class PredictRequest(BaseModel):
    # A strict Pydantic model incoming financial time-series data
    # Expects exactly 20 floats representing preprocessed log-returns.
    # conlist ensures the list has exactly 20 items.
    features: conlist(float, min_length=20, max_length=20) = Field(
        ..., 
        description="Array of 20 floats representing preprocessed log-returns"
    )

class PredictResponse(BaseModel):
    task_id: str = Field(
        ...,
        description="Celery ID for the asynchronous Quantum SVM calculation"
    )

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None

