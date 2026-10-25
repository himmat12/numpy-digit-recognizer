import numpy as np

from digit_recognizer.activations import relu, relu_derivative, softmax


def test_relu_zeroes_negatives():
    z = np.array([[-2.0, 0.0, 3.0]])
    assert np.array_equal(relu(z), [[0.0, 0.0, 3.0]])


def test_relu_derivative():
    z = np.array([[-2.0, 0.0, 3.0]])
    assert np.array_equal(relu_derivative(z), [[0.0, 0.0, 1.0]])


def test_softmax_rows_sum_to_one():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])  # large values must not overflow
    probs = softmax(z)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], 1 / 3)
