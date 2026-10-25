# Step 6: Saving, loading & predicting

Files: [`digit_recognizer/network.py`](../digit_recognizer/network.py),
[`predict.py`](../predict.py)

## 6.1 Saving and loading the model

A trained network is just its four arrays: `W1`, `b1`, `W2`, `b2`.

- `save()` (network.py line 93) writes them to `models/mnist_model.npz` with `np.savez`.
- `load()` (line 99) reads them back. It works out the layer sizes from the array shapes,
  so you don't need to remember them.

This means you train once and can then predict as often as you like without retraining.

## 6.2 Making a prediction: `predict()` (network.py line 87)

```python
np.argmax(self.forward(x), axis=1)
```

Run the forward pass, then pick the digit with the highest probability.

## 6.3 `predict.py`

```bash
python predict.py             # 3 random test images
python predict.py --count 10  # 10 images
```

For each random test image it:

1. draws the image in the terminal as ASCII art (`image_to_ascii()`, line 17), mapping
   each pixel's brightness to a character from `" .:-=+*#%@"`;
2. runs the network and prints the predicted digit, how confident the network is, and the
   true label.

Example:

```
       *@      *@@
       *@     *@*@
       :@.  :*@:-@
        *@@*@@- *@
                @#
                @*
Predicted: 9 (98.9% sure)  True label: 9  -> correct
```

Note `x_test[index : index + 1]`: slicing (instead of `x_test[index]`) keeps the shape
`(1, 784)`. The network always expects a *batch* of images, even when the batch holds just one.

## 6.4 Recap: the whole pipeline

```
data.py        load_mnist()       -> images (n, 784), labels
network.py     forward()          -> probabilities (n, 10)
               loss()             -> how wrong
               backward()         -> gradients
               update()           -> better weights
train.py       train()            -> repeat over mini-batches & epochs
               save()             -> models/mnist_model.npz
predict.py     load() + predict() -> digits!
```

## Ideas to extend the project

- **Add a second hidden layer** (784 → 128 → 64 → 10). You'll need `W3`/`b3` and one more
  step in `forward` and `backward`.
- **Learning-rate decay:** shrink the learning rate a little each epoch.
- **Validation split:** hold out 5,000 training images to tune hyperparameters, so the test
  set stays untouched until the end.
- **Momentum or Adam:** smarter update rules that usually train faster.
- **Your own handwriting:** draw a digit, resize it to 28×28 (white digit on black background,
  like MNIST), scale it to 0..1, and pass it to `predict()`. The API in the next step does
  exactly this for uploaded images.

**Next:** [Step 7: Serving the model with FastAPI](07-serving-api.md)
