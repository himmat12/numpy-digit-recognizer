"""Request and response formats of the API."""

from typing import Annotated

from pydantic import BaseModel, Field

from digit_recognizer.data import IMAGE_SIZE, NUM_CLASSES

Pixel = Annotated[float, Field(ge=0.0, le=1.0)]


class PixelsRequest(BaseModel):
    """A 28x28 image flattened row by row into 784 values between 0 (background) and 1 (ink)."""

    pixels: list[Pixel] = Field(min_length=IMAGE_SIZE, max_length=IMAGE_SIZE)


class PredictionResponse(BaseModel):
    digit: int = Field(description="The most likely digit (0-9).")
    confidence: float = Field(description="Probability of that digit.")
    probabilities: list[float] = Field(
        min_length=NUM_CLASSES, max_length=NUM_CLASSES, description="Probability of each digit 0-9."
    )
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
