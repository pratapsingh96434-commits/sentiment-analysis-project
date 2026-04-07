# app/schema.py
# This file defines the "shape" of your API requests and responses.
# Pydantic validates that incoming data is correct BEFORE it hits your logic.

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Defines what the client must send to /predict.
    These 4 features match the Iris dataset columns.
    Field() lets us add validation (min/max) and documentation.
    """
    sepal_length: float = Field(..., gt=0, lt=20, example=5.1,
                                 description="Sepal length in cm")
    sepal_width:  float = Field(..., gt=0, lt=20, example=3.5,
                                 description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, lt=20, example=1.4,
                                 description="Petal length in cm")
    petal_width:  float = Field(..., gt=0, lt=20, example=0.2,
                                 description="Petal width in cm")

    class Config:
        # Allows Swagger docs to show a filled example
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width":  3.5,
                "petal_length": 1.4,
                "petal_width":  0.2,
            }
        }


class PredictionResponse(BaseModel):
    """
    Defines what the API will return after a prediction.
    Clients always get a predictable, documented response shape.
    """
    prediction:   int   = Field(..., description="Numeric class: 0, 1, or 2")
    species:      str   = Field(..., description="Iris species name")
    confidence:   float = Field(..., description="Model confidence (0.0 – 1.0)")
    status:       str   = Field(default="success")


class HealthResponse(BaseModel):
    """Response shape for the /health endpoint."""
    status:       str
    model_loaded: bool
    version:      str
    message:      Optional[str] = None
