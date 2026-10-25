"""Train the network on MNIST.

Run from the project root:
    python -m digit_recognizer.train
    python -m digit_recognizer.train --epochs 5 --lr 0.05
"""

import argparse
from pathlib import Path

import numpy as np

from digit_recognizer.data import IMAGE_SIZE, NUM_CLASSES, load_mnist, one_hot
from digit_recognizer.network import NeuralNetwork

# Default settings ("hyperparameters").
HIDDEN_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1
SEED = 42

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mnist_model.npz"


def accuracy(network: NeuralNetwork, x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of images whose predicted digit matches the true label."""
    return float(np.mean(network.predict(x) == y))


def train(
    network: NeuralNetwork,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = SEED,
) -> None:
    """Mini-batch gradient descent: repeat forward -> loss -> backward -> update."""
    rng = np.random.default_rng(seed)
    y_train_one_hot = one_hot(y_train)
    num_samples = x_train.shape[0]

    for epoch in range(1, epochs + 1):
        # Shuffle so every epoch sees the batches in a different order.
        order = rng.permutation(num_samples)
        total_loss = 0.0

        for start in range(0, num_samples, batch_size):
            batch = order[start : start + batch_size]
            x_batch, y_batch = x_train[batch], y_train_one_hot[batch]

            probs = network.forward(x_batch)
            total_loss += network.loss(probs, y_batch) * len(batch)
            grads = network.backward(probs, y_batch)
            network.update(grads, learning_rate)

        avg_loss = total_loss / num_samples
        test_acc = accuracy(network, x_test, y_test)
        print(f"Epoch {epoch:2d}/{epochs}  loss: {avg_loss:.4f}  test accuracy: {test_acc:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a digit recognizer on MNIST.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading MNIST ...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Train: {x_train.shape[0]} images, Test: {x_test.shape[0]} images\n")

    network = NeuralNetwork(IMAGE_SIZE, args.hidden_size, NUM_CLASSES, seed=SEED)
    train(network, x_train, y_train, x_test, y_test, args.epochs, args.batch_size, args.lr)

    network.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
