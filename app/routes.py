# app/routes.py
# This file defines your API endpoints (routes).
# Each function handles ONE HTTP request. Routes don't know HOW the model works —
# they just call model.py and return a response. Clean separation!

import logging
from fastapi import APIRouter, HTTPException

from app.schema import PredictionRequest, PredictionResponse, HealthResponse
from app import model as ml_model

logger = logging.getLogger(__name__)

# APIRouter is like a "mini FastAPI app" — we register it in main.py
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API and ML model are running correctly.",
    tags=["Monitoring"],
)
def health_check():
    """
    GET /health
    Returns the API status and whether the model is loaded.
    Use this in AWS ECS / EKS health probes or uptime monitors.
    """
    loaded = ml_model.is_model_loaded()

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        version="1.0.0",
        message="All systems operational." if loaded else "Model not loaded!",
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Iris Species Prediction",
    description="Send 4 iris flower measurements and get the predicted species.",
    tags=["ML Prediction"],
)
def predict(request: PredictionRequest):
    """
    POST /predict
    Accepts iris measurements, returns predicted species + confidence.

    FastAPI automatically:
    - Validates the request body against PredictionRequest schema
    - Returns a 422 error if any field is missing or out of range
    - Generates Swagger documentation
    """
    logger.info(f"Prediction request received: {request.model_dump()}")

    # Build the feature list in the same order the model was trained on
    features = [
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width,
    ]

    try:
        prediction_class, species, confidence = ml_model.predict(features)
    except RuntimeError as e:
        # Model not loaded — return 503 Service Unavailable
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Any other unexpected error — return 500
        logger.exception(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

    return PredictionResponse(
        prediction=prediction_class,
        species=species,
        confidence=round(confidence, 4),
        status="success",
    )
