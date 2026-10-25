"""Activation functions used by the network."""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """Keep positive values, replace negatives with 0."""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Slope of ReLU: 1 where z > 0, otherwise 0."""
    return (z > 0).astype(z.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    """Turn each row of raw scores into probabilities that sum to 1."""
    # Subtracting the row max does not change the result but avoids overflow in exp().
    shifted = z - z.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)
