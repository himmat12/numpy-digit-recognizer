# Step 5: The training loop

File: [`digit_recognizer/train.py`](../digit_recognizer/train.py)

## 5.1 Hyperparameters (lines 17–21)

These are settings *we* choose. The network does not learn them.

| Name            | Default | Meaning                                              |
|-----------------|---------|------------------------------------------------------|
| `HIDDEN_SIZE`   | 128     | neurons in the hidden layer                          |
| `EPOCHS`        | 10      | full passes over the 60,000 training images          |
| `BATCH_SIZE`    | 64      | images used for each weight update                   |
| `LEARNING_RATE` | 0.1     | size of each gradient-descent step                   |
| `SEED`          | 42      | makes results reproducible                           |

All except `SEED` can be overridden from the command line (see `parse_args()`, line 66):

```bash
python -m digit_recognizer.train --epochs 5 --lr 0.05 --batch-size 128 --hidden-size 64
```

## 5.2 Mini-batches

We don't update weights after every single image, which is noisy and slow. We also don't wait
for all 60,000, which is slow per step. Instead we use **mini-batches** of 64 images, a good
middle ground. One epoch is therefore 60000 / 64 ≈ 938 updates.

## 5.3 `train()` (line 31)

```python
for epoch in range(1, epochs + 1):
    order = rng.permutation(num_samples)          # 1. shuffle
    for start in range(0, num_samples, batch_size):
        batch = order[start : start + batch_size] # 2. take the next 64 images
        probs = network.forward(x_batch)          # 3. predict
        loss  = network.loss(probs, y_batch)      # 4. measure the error
        grads = network.backward(probs, y_batch)  # 5. compute gradients
        network.update(grads, learning_rate)      # 6. improve the weights
```

Steps 3–6 are the core of almost every neural-network training loop, including the ones
inside PyTorch and TensorFlow.

**Why shuffle?** Without it, the network would see the batches in the same order every epoch
and could pick up patterns from that order instead of from the digits.

After each epoch we print the average training loss and the **test accuracy**
(`accuracy()`, line 26). Test images are never used for training, so this number shows how
well the network handles new handwriting.

## 5.4 Expected output

```
Epoch  1/10  loss: 0.3676  test accuracy: 93.63%
Epoch  2/10  loss: 0.1962  test accuracy: 94.80%
...
Epoch 10/10  loss: 0.0551  test accuracy: 97.67%
Model saved to .../models/mnist_model.npz
```

The loss falls and the accuracy rises. That is the network learning.

## 5.5 `main()` (line 75)

`main()` connects everything: parse the arguments, load the data, build the network, train it,
save it. The `if __name__ == "__main__":` guard means this runs only when the file is executed
directly, not when it's imported.

## Things to try

- `--lr 1.0` (too high: watch the loss misbehave) or `--lr 0.01` (slow).
- `--hidden-size 16` vs `--hidden-size 512`: how does accuracy change?
- `--epochs 30`: does test accuracy keep improving, or level off?

**Next:** [Step 6: Saving, loading & predicting](06-prediction.md)
