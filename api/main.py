"""The FastAPI application.

Run locally from the project root:
    uvicorn api.main:app --reload
Then open http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from api.config import get_settings
from api.preprocessing import InvalidImageError, image_bytes_to_input
from api.schemas import HealthResponse, PixelsRequest, PredictionResponse
from digit_recognizer.network import NeuralNetwork


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and keep it on app.state for every request."""
    settings = get_settings()
    if not settings.model_path.exists():
        raise RuntimeError(f"Model file not found: {settings.model_path}")

    app.state.network = NeuralNetwork.load(settings.model_path)
    app.state.model_version = settings.model_version
    yield


app = FastAPI(
    title="NumPy Digit Recognizer",
    description="Recognises handwritten digits (0-9) with a neural network written in NumPy.",
    version="1.0.0",
    lifespan=lifespan,
)


def make_prediction(request: Request, x: np.ndarray) -> PredictionResponse:
    """Run the network on one input of shape (1, 784) and package the result."""
    probs = request.app.state.network.forward(x)[0]
    digit = int(np.argmax(probs))
    return PredictionResponse(
        digit=digit,
        confidence=float(probs[digit]),
        probabilities=[float(p) for p in probs],
        model_version=request.app.state.model_version,
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Check that the service is up and see which model it is serving."""
    return HealthResponse(status="ok", model_version=request.app.state.model_version)


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """Upload an image of a single digit (PNG, JPG, ...) and get the predicted digit."""
    try:
        x = image_bytes_to_input(await file.read())
    except InvalidImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return make_prediction(request, x)


@app.post("/predict/pixels", response_model=PredictionResponse)
def predict_pixels(request: Request, body: PixelsRequest) -> PredictionResponse:
    """Send 784 pixel values (28x28, row by row, 0..1) and get the predicted digit."""
    x = np.array(body.pixels, dtype=np.float32).reshape(1, -1)
    return make_prediction(request, x)
