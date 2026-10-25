# NumPy Digit Recognizer

A small, readable neural network that learns to recognise handwritten digits (0–9) from the
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. It's built from scratch with
**NumPy only**, so every step (forward pass, backpropagation and gradient descent) is code you
can read. No deep-learning framework is involved.

It reaches about **97.7% test accuracy** in roughly 10 seconds on a laptop CPU.

The model is trained by GitHub Actions, published as a versioned GitHub Release, and served by
a **FastAPI** service packaged as a **Docker** image.

## Quick start

```bash
python3 -m venv .venv                   # 1. create an isolated environment
source .venv/bin/activate               # 2. activate it
pip install -r requirements.txt         # 3. install numpy (see table below for more)

python -m digit_recognizer.train        # 4. download MNIST + train + save model
python predict.py                       # 5. see predictions on test images
```

## Serving the model

Run the published Docker image (it contains the latest model trained by CI):

```bash
docker compose up -d
curl localhost:8000/health
curl -F file=@my_digit.png localhost:8000/predict
```

Interactive API docs: <http://localhost:8000/docs>. Or run the API without Docker:

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

## Dependencies

| File                   | Installs                                        | Use it to...     |
|------------------------|-------------------------------------------------|------------------|
| `requirements.txt`     | numpy                                           | train / predict  |
| `requirements-api.txt` | + fastapi, uvicorn, pillow, python-multipart    | run the API      |
| `requirements-dev.txt` | + pytest, httpx                                 | run the tests    |

## Project layout

```
.
├── digit_recognizer/        the package
│   ├── data.py              download & prepare MNIST
│   ├── activations.py       ReLU and softmax
│   ├── network.py           the NeuralNetwork class (forward, backward, update)
│   └── train.py             training loop + command-line entry point
├── api/                     FastAPI service
│   ├── config.py            settings from environment variables
│   ├── preprocessing.py     uploaded image -> network input
│   ├── schemas.py           request/response formats
│   └── main.py              the app and its endpoints
├── predict.py               try the trained model on test images
├── tests/                   unit tests (network, gradient check, API)
├── docs/                    step-by-step guide (start here!)
├── .github/workflows/
│   ├── train-model.yml      train -> check accuracy -> publish GitHub Release
│   └── build-image.yml      build Docker image with that model -> push to ghcr.io
├── Dockerfile               downloads + verifies the released model, runs the API
├── docker-compose.yml       one-command run of the service
└── requirements*.txt        dependencies (see table above)
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
7. [Serving the model with FastAPI](docs/07-serving-api.md)
8. [Docker & the CI pipeline](docs/08-docker-and-ci.md)

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```
