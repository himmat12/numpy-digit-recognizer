# Step 3: The network & the forward pass

Files: [`digit_recognizer/network.py`](../digit_recognizer/network.py),
[`digit_recognizer/activations.py`](../digit_recognizer/activations.py)

## 3.1 Architecture

```
 input layer        hidden layer          output layer
 784 pixels   ->    128 neurons (ReLU) -> 10 neurons (softmax)
                                           = probability of each digit 0..9
```

Every neuron in one layer is connected to every neuron in the next. This is called a
"fully connected" or "dense" layer. Each connection has a **weight**, and each neuron has a
**bias**. These numbers are what the network learns.

| Parameter | Shape      | Count   |
|-----------|------------|---------|
| `W1`      | (784, 128) | 100,352 |
| `b1`      | (128,)     | 128     |
| `W2`      | (128, 10)  | 1,280   |
| `b2`      | (10,)      | 10      |
|           | **total**  | **101,770** |

## 3.2 Initialisation: `__init__()` (network.py line 21)

Weights start as small random numbers. If they all started equal, every neuron would learn
the same thing. Biases start at zero.

We use **He initialisation** (line 25): random normal values with standard deviation
`sqrt(2 / number_of_inputs)`. This scale keeps signals from shrinking or exploding as they
pass through ReLU layers. A fixed `seed` makes runs reproducible.

## 3.3 Activation functions: `activations.py`

**ReLU** (line 6): `relu(z) = max(0, z)`.
It keeps positive values and turns negatives into 0. Without a non-linear function like this,
stacking layers would be no more powerful than a single layer.

**Softmax** (line 16) turns the 10 raw output scores into probabilities that are all positive
and add up to 1:

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Line 19 subtracts the largest score before `exp()`. The result is mathematically the same,
but it stops `exp(1000)` from overflowing to infinity.

## 3.4 The forward pass: `forward()` (network.py line 35)

The "forward pass" pushes a batch of images through the network:

```python
z1 = x @ W1 + b1      # (n, 784) @ (784, 128) -> (n, 128)   weighted sums
a1 = relu(z1)         #                           (n, 128)   hidden activations
z2 = a1 @ W2 + b2     # (n, 128) @ (128, 10)  -> (n, 10)     output scores
probs = softmax(z2)   #                           (n, 10)    probabilities
```

`@` is matrix multiplication. Because `x` holds **n images at once** (one per row), the whole
batch is processed in a single step. That is much faster than looping over images.

`forward()` also stores `x`, `z1` and `a1` in `self._cache`. The backward pass needs these
values (next step).

## Try it

```python
import numpy as np
from digit_recognizer import NeuralNetwork
net = NeuralNetwork(784, 128, 10)
probs = net.forward(np.random.rand(2, 784))
print(probs.shape, probs.sum(axis=1))   # (2, 10) [1. 1.]
```

An untrained network gives roughly 10% to every digit. It is just guessing.

**Next:** [Step 4: Loss & backpropagation](04-backpropagation.md)
