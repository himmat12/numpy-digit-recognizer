"""Turn an uploaded image into the input format the network expects."""

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIDE = 28  # MNIST images are 28x28


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a readable image."""


def image_bytes_to_input(data: bytes) -> np.ndarray:
    """Convert image file bytes (PNG, JPG, ...) to an array of shape (1, 784) with values 0..1.

    Steps:
      1. open the image and convert it to grayscale
      2. resize it to 28x28
      3. scale pixel values to 0..1
      4. invert if the background is light: MNIST digits are light ink on a dark background
    """
    try:
        image = Image.open(io.BytesIO(data)).convert("L")
    except (UnidentifiedImageError, OSError) as error:
        raise InvalidImageError("File is not a valid image.") from error

    image = image.resize((IMAGE_SIDE, IMAGE_SIDE), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    if pixels.mean() > 0.5:  # mostly light, e.g. dark pen on white paper
        pixels = 1.0 - pixels

    return pixels.reshape(1, IMAGE_SIDE * IMAGE_SIDE)
