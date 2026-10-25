"""Load the trained model and show its predictions on a few test images.

Run from the project root (after training):
    python predict.py
    python predict.py --count 10
"""

import argparse

import numpy as np

from digit_recognizer.data import load_mnist
from digit_recognizer.network import NeuralNetwork
from digit_recognizer.train import MODEL_PATH


def image_to_ascii(image: np.ndarray) -> str:
    """Draw a 784-pixel image as 28 lines of text so it can be seen in the terminal."""
    shades = " .:-=+*#%@"  # light -> dark
    rows = image.reshape(28, 28)
    return "\n".join(
        "".join(shades[int(pixel * (len(shades) - 1))] for pixel in row) for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show predictions on random test images.")
    parser.add_argument("--count", type=int, default=3, help="number of images to show")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run: python -m digit_recognizer.train")

    network = NeuralNetwork.load(MODEL_PATH)
    _, _, x_test, y_test = load_mnist()

    rng = np.random.default_rng()
    for index in rng.choice(len(x_test), size=args.count, replace=False):
        image = x_test[index : index + 1]  # keep the batch dimension: shape (1, 784)
        probs = network.forward(image)[0]
        predicted = int(np.argmax(probs))

        print(image_to_ascii(image[0]))
        verdict = "correct" if predicted == y_test[index] else "WRONG"
        print(
            f"Predicted: {predicted} ({probs[predicted]:.1%} sure)  "
            f"True label: {y_test[index]}  -> {verdict}\n"
        )


if __name__ == "__main__":
    main()
