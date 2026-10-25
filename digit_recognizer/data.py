"""Download, load and prepare the MNIST handwritten digit dataset.

MNIST has 70,000 grayscale images of digits (0-9), each 28x28 pixels:
60,000 for training and 10,000 for testing.
"""

import gzip
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Mirrors are tried in order, in case one is unavailable.
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

IMAGE_SIZE = 28 * 28  # 784 pixels per image
NUM_CLASSES = 10  # digits 0-9


def download_file(filename: str) -> Path:
    """Download one MNIST file into DATA_DIR (skipped if it already exists)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    if path.exists():
        return path

    for mirror in MIRRORS:
        try:
            print(f"Downloading {filename} from {mirror} ...")
            urllib.request.urlretrieve(mirror + filename, path)
            return path
        except OSError as error:
            print(f"  failed: {error}")
            path.unlink(missing_ok=True)

    raise RuntimeError(f"Could not download {filename} from any mirror.")


def read_images(path: Path) -> np.ndarray:
    """Read an IDX image file into an array of shape (num_images, 784)."""
    with gzip.open(path, "rb") as f:
        # The first 16 bytes are a header (magic number, count, rows, cols).
        pixels = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return pixels.reshape(-1, IMAGE_SIZE)


def read_labels(path: Path) -> np.ndarray:
    """Read an IDX label file into an array of shape (num_labels,)."""
    with gzip.open(path, "rb") as f:
        # The first 8 bytes are a header (magic number, count).
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale pixel values from 0..255 to 0.0..1.0."""
    return images.astype(np.float32) / 255.0


def one_hot(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Turn labels like [3, 0] into rows like [0,0,0,1,0,...] and [1,0,0,0,0,...]."""
    encoded = np.zeros((labels.size, num_classes), dtype=np.float32)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_test, y_test).

    x_* are normalised images of shape (n, 784).
    y_* are integer labels of shape (n,).
    """
    paths = {name: download_file(filename) for name, filename in FILES.items()}

    x_train = normalize(read_images(paths["train_images"]))
    y_train = read_labels(paths["train_labels"])
    x_test = normalize(read_images(paths["test_images"]))
    y_test = read_labels(paths["test_labels"])

    return x_train, y_train, x_test, y_test
