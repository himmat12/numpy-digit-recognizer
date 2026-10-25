import numpy as np

from digit_recognizer.data import one_hot
from digit_recognizer.network import NeuralNetwork


def make_batch(n=8, input_size=5, num_classes=3, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, input_size))
    y = one_hot(rng.integers(0, num_classes, size=n), num_classes)
    return x, y


def test_forward_output_shape_and_probabilities():
    network = NeuralNetwork(5, 4, 3)
    x, _ = make_batch()
    probs = network.forward(x)
    assert probs.shape == (8, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_backward_matches_numerical_gradient():
    """Compare backprop gradients with a slow but simple finite-difference estimate."""
    network = NeuralNetwork(5, 4, 3)
    x, y = make_batch()

    grads = network.backward(network.forward(x), y)

    eps = 1e-5
    for name in ["W1", "b1", "W2", "b2"]:
        param = getattr(network, name)
        numerical = np.zeros_like(param)
        for i in np.ndindex(param.shape):
            original = param[i]
            param[i] = original + eps
            loss_plus = network.loss(network.forward(x), y)
            param[i] = original - eps
            loss_minus = network.loss(network.forward(x), y)
            param[i] = original
            numerical[i] = (loss_plus - loss_minus) / (2 * eps)

        assert np.allclose(grads[name], numerical, atol=1e-6), name


def test_training_reduces_loss():
    network = NeuralNetwork(5, 16, 3)
    x, y = make_batch(n=32)

    first_loss = network.loss(network.forward(x), y)
    for _ in range(200):
        network.update(network.backward(network.forward(x), y), learning_rate=0.1)
    final_loss = network.loss(network.forward(x), y)

    assert final_loss < first_loss / 2


def test_save_and_load(tmp_path):
    network = NeuralNetwork(5, 4, 3)
    path = tmp_path / "model.npz"
    network.save(path)

    loaded = NeuralNetwork.load(path)
    x, _ = make_batch()
    assert np.allclose(network.forward(x), loaded.forward(x))
