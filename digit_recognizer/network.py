"""A two-layer neural network: input -> hidden (ReLU) -> output (softmax)."""

from pathlib import Path

import numpy as np

from digit_recognizer.activations import relu, relu_derivative, softmax


class NeuralNetwork:
    """A small fully connected network for classification.

    Shapes (n = number of images in a batch):
        x  : (n, input_size)          input images
        W1 : (input_size, hidden_size) weights of the hidden layer
        b1 : (hidden_size,)            biases of the hidden layer
        W2 : (hidden_size, output_size) weights of the output layer
        b2 : (output_size,)            biases of the output layer
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)

        # He initialisation: random weights scaled by sqrt(2 / inputs), which suits ReLU.
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

        # Values saved during forward() that backward() needs.
        self._cache: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- forward

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute class probabilities for a batch of images. Returns shape (n, output_size)."""
        z1 = x @ self.W1 + self.b1  # hidden layer raw scores
        a1 = relu(z1)  # hidden layer activations
        z2 = a1 @ self.W2 + self.b2  # output layer raw scores
        probs = softmax(z2)  # output probabilities

        self._cache = {"x": x, "z1": z1, "a1": a1}
        return probs

    # ------------------------------------------------------------------- loss

    @staticmethod
    def loss(probs: np.ndarray, y_one_hot: np.ndarray) -> float:
        """Average cross-entropy: how 'surprised' the network is by the true labels."""
        eps = 1e-12  # avoid log(0)
        return float(-np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=1)))

    # --------------------------------------------------------------- backward

    def backward(self, probs: np.ndarray, y_one_hot: np.ndarray) -> dict[str, np.ndarray]:
        """Backpropagation: compute the gradient of the loss for every parameter.

        Must be called right after forward() on the same batch.
        """
        x, z1, a1 = self._cache["x"], self._cache["z1"], self._cache["a1"]
        n = x.shape[0]

        # Output layer. For softmax + cross-entropy the gradient is simply (probs - targets).
        dz2 = (probs - y_one_hot) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        # Hidden layer. Push the error back through W2, then through ReLU.
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ----------------------------------------------------------------- update

    def update(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        """Gradient descent: nudge each parameter a small step against its gradient."""
        self.W1 -= learning_rate * grads["W1"]
        self.b1 -= learning_rate * grads["b1"]
        self.W2 -= learning_rate * grads["W2"]
        self.b2 -= learning_rate * grads["b2"]

    # ---------------------------------------------------------------- predict

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the most likely digit for each image. Shape (n,)."""
        return np.argmax(self.forward(x), axis=1)

    # ------------------------------------------------------------- save/load

    def save(self, path: Path) -> None:
        """Save the weights to a .npz file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: Path) -> "NeuralNetwork":
        """Create a network from weights saved with save()."""
        params = np.load(path)
        input_size, hidden_size = params["W1"].shape
        output_size = params["W2"].shape[1]

        network = cls(input_size, hidden_size, output_size)
        network.W1, network.b1 = params["W1"], params["b1"]
        network.W2, network.b2 = params["W2"], params["b2"]
        return network
