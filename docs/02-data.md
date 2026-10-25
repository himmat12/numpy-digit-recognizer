# Step 2: Data: loading and preparing MNIST

File: [`digit_recognizer/data.py`](../digit_recognizer/data.py)

## What MNIST looks like

MNIST contains 70,000 grayscale images of handwritten digits:

- **60,000 training images**, which the network learns from;
- **10,000 test images**, which it never trains on and are used to measure real accuracy.

Each image is 28×28 pixels. Each pixel is a number from 0 (empty background) to 255 (full ink).
Each image has a label telling you which digit it shows (0–9).

## 2.1 Downloading: `download_file()` (line 32)

The four MNIST files are listed in `FILES` (line 21). `download_file()` saves each one to
`data/` using Python's built-in `urllib`, so no extra dependency is needed.
If a file already exists, the download is skipped. If one mirror fails, the next in `MIRRORS`
(line 16) is tried.

## 2.2 Reading the files: `read_images()` / `read_labels()` (lines 51, 59)

The files are gzip-compressed and use the simple **IDX** binary format: a small header
followed by raw bytes.

- Image files have a 16-byte header, then one byte per pixel.
- Label files have an 8-byte header, then one byte per label.

`np.frombuffer(..., offset=16)` skips the header and turns the bytes into an array. Then
`.reshape(-1, 784)` puts each image on its own row: the 28×28 grid is flattened into
784 numbers, because our network takes a flat list of inputs.

```
image (28×28)   ->   one row of 784 numbers
```

## 2.3 Normalising: `normalize()` (line 66)

Pixel values are divided by 255, so they range from **0.0 to 1.0** instead of 0 to 255.
Small, consistent input values keep the numbers inside the network in a reasonable range.
This makes training much more stable.

## 2.4 One-hot encoding: `one_hot()` (line 71)

The network outputs 10 probabilities, one per digit. To compare those with the true label,
we turn each label into a vector of the same shape:

```
label 3  ->  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
label 0  ->  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

The line `encoded[np.arange(n), labels] = 1.0` sets one "1" per row in a single step.

## 2.5 Putting it together: `load_mnist()` (line 78)

This returns four arrays:

| Array     | Shape         | Meaning                       |
|-----------|---------------|-------------------------------|
| `x_train` | (60000, 784)  | normalised training images    |
| `y_train` | (60000,)      | training labels (integers)    |
| `x_test`  | (10000, 784)  | normalised test images        |
| `y_test`  | (10000,)      | test labels (integers)        |

Labels stay as integers here. They are one-hot encoded only where they're needed (in training).

## Try it

```python
from digit_recognizer.data import load_mnist
x_train, y_train, x_test, y_test = load_mnist()
print(x_train.shape, y_train[:10])
```

**Next:** [Step 3: The network & forward pass](03-network.md)
