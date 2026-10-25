# Step 4: Loss & backpropagation

File: [`digit_recognizer/network.py`](../digit_recognizer/network.py)

Training means: *change the weights so the predictions get better*. To do that we need
(1) a number that measures "how wrong" the network is, and
(2) for each weight, the direction that makes that number smaller.

## 4.1 Loss: `loss()` (line 48)

We use **cross-entropy loss**:

```
loss = -mean( log(probability given to the correct digit) )
```

- If the network gives the correct digit probability 1.0, then log(1) = 0, so the loss is 0. Perfect.
- If it gives the correct digit probability 0.01, then -log(0.01) ≈ 4.6. Very wrong.

Multiplying by the one-hot label picks out the correct digit's probability in each row.
A tiny `eps` keeps us from taking `log(0)`.

## 4.2 Gradients: the idea

A **gradient** tells you how much the loss would change if you nudged a parameter slightly.
If increasing a weight raises the loss, we should decrease that weight, and the other way round.

**Backpropagation** computes all gradients efficiently with the chain rule. It starts at the
output and works backwards, layer by layer.

## 4.3 `backward()` step by step (line 55)

**Output layer** (line 64):

```python
dz2 = (probs - y_one_hot) / n
```

This is the key result: for softmax combined with cross-entropy, the gradient with respect to
the output scores is simply *prediction minus target*. For example, if the true digit is 3 and
the network said 0.7 for it, the error there is `0.7 - 1 = -0.3`, which means "raise this
score". Dividing by `n` averages over the batch.

```python
dW2 = a1.T @ dz2       # each weight's gradient = its input × the error it fed into
db2 = dz2.sum(axis=0)  # a bias's input is always 1, so just sum the errors
```

**Hidden layer** (line 69):

```python
da1 = dz2 @ W2.T                  # send the error back through the output weights
dz1 = da1 * relu_derivative(z1)   # ReLU only lets error through where z1 > 0
dW1 = x.T @ dz1
db1 = dz1.sum(axis=0)
```

The pattern is the same as in the output layer. The only new part is multiplying by ReLU's
slope: 1 for neurons that were active, 0 for neurons that were switched off.

Each gradient has exactly the same shape as its parameter (`dW1` matches `W1`, and so on).

## 4.4 Update: `update()` (line 78)

**Gradient descent** moves every parameter a small step *against* its gradient:

```python
W1 -= learning_rate * dW1
```

The **learning rate** (0.1 by default) controls the step size. Too large and training
jumps around. Too small and it crawls.

## 4.5 How do we know backprop is correct?

`tests/test_network.py::test_backward_matches_numerical_gradient` checks every gradient
against a slow brute-force estimate. It nudges each parameter by ±0.00001, measures how the
loss changes, and compares the result with `backward()`. If the two match, the math is right.
This **gradient check** is the standard way to test hand-written backprop.

**Next:** [Step 5: Training loop](05-training.md)
