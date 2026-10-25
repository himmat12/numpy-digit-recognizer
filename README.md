# NumPy Digit Recognizer

A small, readable neural network that learns to recognise handwritten digits (0–9) from the
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. It's built from scratch with
**NumPy only**, so every step (forward pass, backpropagation and gradient descent) is code you
can read. No deep-learning framework is involved.

It reaches about **97.7% test accuracy** in roughly 10 seconds on a laptop CPU.

## Quick start

```bash
python3 -m venv .venv                   # 1. create an isolated environment
source .venv/bin/activate               # 2. activate it
pip install -r requirements.txt         # 3. install numpy (add -dev for pytest)

python -m digit_recognizer.train        # 4. download MNIST + train + save model
python predict.py                       # 5. see predictions on test images
```

## Project layout

```
.
├── digit_recognizer/        the package
│   ├── data.py              download & prepare MNIST
│   ├── activations.py       ReLU and softmax
│   ├── network.py           the NeuralNetwork class (forward, backward, update)
│   └── train.py             training loop + command-line entry point
├── predict.py               try the trained model on test images
├── tests/                   unit tests, including a gradient check
├── docs/                    step-by-step guide (start here!)
├── requirements.txt         runtime dependencies (numpy)
└── requirements-dev.txt     + development dependencies (pytest)
```

`data/` (downloaded dataset), `models/` (trained weights) and `.venv/` are generated
automatically and are git-ignored.

## Step-by-step guide

Read these in order. Each page explains one part of the code.

1. [Setup: virtual environment & dependencies](docs/01-setup.md)
2. [Data: loading and preparing MNIST](docs/02-data.md)
3. [The network & the forward pass](docs/03-network.md)
4. [Loss & backpropagation](docs/04-backpropagation.md)
5. [Training loop](docs/05-training.md)
6. [Saving, loading & predicting](docs/06-prediction.md)

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```
