# app/model.py
# This file is responsible for ONE thing: loading the ML model and making predictions.
# It is completely separated from FastAPI so you can test it independently.

import pickle
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

# Maps numeric predictions → human-readable species names
SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

# Module-level variable — model is loaded ONCE at startup, not on every request
_model = None


def load_model() -> None:
    """
    Loads the model from disk into memory.
    Called once when the app starts (see main.py lifespan).
    Raises FileNotFoundError if model.pkl is missing.
    """
    global _model

    # Support running from project root OR from the app/ directory
    possible_paths = [
        os.path.join("model", "model.pkl"),
        os.path.join("..", "model", "model.pkl"),
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError(
            "model.pkl not found. Run `python train_model.py` first."
        )

    with open(model_path, "rb") as f:
        _model = pickle.load(f)

    logger.info(f"Model loaded successfully from: {model_path}")


def is_model_loaded() -> bool:
    """Returns True if the model is ready to use."""
    return _model is not None


def predict(features: list) -> Tuple[int, str, float]:
    """
    Runs a prediction given a list of 4 float features.

    Args:
        features: [sepal_length, sepal_width, petal_length, petal_width]

    Returns:
        (prediction_class, species_name, confidence_score)

    Raises:
        RuntimeError: If model isn't loaded yet.
    """
    if _model is None:
        raise RuntimeError("Model is not loaded. Cannot make predictions.")

    # sklearn expects a 2D array: [[f1, f2, f3, f4]]
    input_data = [features]

    prediction_class = int(_model.predict(input_data)[0])

    # predict_proba returns probabilities for each class
    # e.g. [[0.02, 0.05, 0.93]] — we take the max as confidence
    probabilities = _model.predict_proba(input_data)[0]
    confidence    = float(max(probabilities))

    species = SPECIES_MAP.get(prediction_class, "unknown")

    logger.info(
        f"Prediction: class={prediction_class}, species={species}, "
        f"confidence={confidence:.4f}"
    )

    return prediction_class, species, confidence
