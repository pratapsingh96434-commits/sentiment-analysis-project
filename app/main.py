# app/main.py
# This is the ENTRY POINT of your application.
# Think of it as the "boss file" — it:
#   1. Creates the FastAPI app
#   2. Configures logging
#   3. Loads the ML model on startup
#   4. Registers all routes
#   5. Starts the server

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import model as ml_model
from app.routes import router

# ── Logging Configuration ────────────────────────────────────────────────────
# All log messages will show: timestamp | level | file | message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),          # Print to console
        logging.FileHandler("app.log", mode="a"),   # Write to file
    ],
)
logger = logging.getLogger(__name__)


# ── Lifespan (Startup & Shutdown) ────────────────────────────────────────────
# This runs BEFORE the first request and AFTER the last request.
# Loading the model here means it's loaded once, not on every API call.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Starting ML Prediction API...")
    try:
        ml_model.load_model()
        logger.info("Startup complete. API is ready.")
    except FileNotFoundError as e:
        logger.error(f"STARTUP FAILED: {e}")
        logger.error("Run `python train_model.py` to generate model.pkl first.")
    yield
    # SHUTDOWN (runs when server stops)
    logger.info("Shutting down ML Prediction API.")


# ── FastAPI App Instance ─────────────────────────────────────────────────────
app = FastAPI(
    title="ML Prediction API",
    description=(
        "A production-style FastAPI application that serves an Iris species "
        "classifier. Built with FastAPI, scikit-learn, and Pydantic."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI  → http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc UI    → http://localhost:8000/redoc
)


# ── CORS Middleware ──────────────────────────────────────────────────────────
# Allows frontend apps (React, etc.) to call your API.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # e.g. ["https://myapp.com"] in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register Routes ──────────────────────────────────────────────────────────
# All routes from routes.py are available under /api/v1/
app.include_router(router, prefix="/api/v1")


# ── Root Endpoint ────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    """Welcome message. Visit /docs for the full API documentation."""
    return {
        "message": "ML Prediction API is running!",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/api/v1/health",
        "predict": "http://localhost:8000/api/v1/predict",
    }
