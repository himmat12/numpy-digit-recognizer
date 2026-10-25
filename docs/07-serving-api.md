# Step 7: Serving the model with FastAPI

Folder: [`api/`](../api/)

So far the model only runs from the command line. A web **API** lets any program (a web page,
a phone app, another service) send a digit and get a prediction back over HTTP.
We use [FastAPI](https://fastapi.tiangolo.com/). It is small, fast, and generates interactive
documentation automatically.

## 7.1 How the `api/` package is organised

| File               | Job                                                        |
|--------------------|------------------------------------------------------------|
| `config.py`        | read settings (model path, model version) from env vars    |
| `preprocessing.py` | turn an uploaded image into the network's input            |
| `schemas.py`       | define what requests and responses look like               |
| `main.py`          | the FastAPI app and its endpoints                          |

The API reuses the existing code: it loads weights with `NeuralNetwork.load()` and calls
`forward()`. None of the network code changed.

## 7.2 Settings: `config.py`

`get_settings()` (line 16) reads two **environment variables**:

| Variable        | Default                     | Meaning                                |
|-----------------|-----------------------------|----------------------------------------|
| `MODEL_PATH`    | `models/mnist_model.npz`    | where the trained weights are          |
| `MODEL_VERSION` | `local`                     | label shown in responses, e.g. `model-v3` |

Environment variables let the same code run on your laptop, in tests, and in Docker,
each with different settings and no code changes.

## 7.3 Loading the model once: `lifespan` (main.py line 20)

Loading weights from disk for every request would be wasteful. FastAPI's `lifespan` function
runs **once at startup**. It loads the network and stores it on `app.state`, where every
request can reach it. If the model file is missing, the service refuses to start, which is
better than starting and then failing every request.

## 7.4 Endpoints

| Method | Path              | Input                         | Purpose                         |
|--------|-------------------|-------------------------------|---------------------------------|
| GET    | `/health`         | none                          | is it up? which model?          |
| POST   | `/predict`        | image file (PNG, JPG, ...)    | predict from a picture          |
| POST   | `/predict/pixels` | JSON: 784 numbers from 0 to 1 | predict from raw pixels         |
| GET    | `/docs`           | none                          | interactive docs (try it in the browser) |

Both predict endpoints return the same shape (`PredictionResponse` in `schemas.py`, line 18):

```json
{
  "digit": 7,
  "confidence": 0.998,
  "probabilities": [0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.998, 0.0, 0.001],
  "model_version": "model-v1"
}
```

Both endpoints share one helper, `make_prediction()` (main.py line 39), so the logic is not
duplicated.

## 7.5 Validation: `schemas.py`

`PixelsRequest` (line 12) says: "a list of **exactly 784** numbers, each between 0 and 1".
FastAPI checks every request against this. Bad input gets an automatic **422** error with a
clear message, and your code never sees it.

For uploads, if the file isn't an image, `preprocessing.py` raises `InvalidImageError`.
The endpoint turns it into a **400** error.

## 7.6 Preprocessing an image: `image_bytes_to_input()` (preprocessing.py line 15)

A photo of a digit looks nothing like an MNIST image, so we convert it:

1. **Grayscale.** Colour doesn't matter, and MNIST is grayscale.
2. **Resize to 28×28.** The network has exactly 784 inputs.
3. **Scale to 0..1.** This matches how the training data was normalised (step 2).
4. **Invert if the background is light.** MNIST digits are *light ink on a dark background*.
   A digit drawn with a dark pen on white paper would confuse the network, so we flip it
   when the average pixel is bright.

For best results, send one centred digit that fills most of the image.

## 7.7 Run it locally

```bash
pip install -r requirements-api.txt      # or requirements-dev.txt for tests too
python -m digit_recognizer.train         # if you don't have models/mnist_model.npz yet
uvicorn api.main:app --reload            # --reload restarts on code changes
```

Open <http://localhost:8000/docs>, choose an endpoint, and click "Try it out".
Or use `curl`:

```bash
curl localhost:8000/health
curl -F file=@my_digit.png localhost:8000/predict
```

## 7.8 Tests: `tests/test_api.py`

The tests don't need a trained model. A fixture saves a tiny random network to a temporary
folder and points `MODEL_PATH` at it. Then FastAPI's `TestClient` calls the endpoints in
memory, with no real server, and checks the response shapes and error codes. We're testing
the API plumbing here, not the model's accuracy.

**Next:** [Step 8: Docker & the CI pipeline](08-docker-and-ci.md)
