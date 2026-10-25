# Step 8: Docker & the CI pipeline

Files: [`.github/workflows/`](../.github/workflows/), [`Dockerfile`](../Dockerfile),
[`docker-compose.yml`](../docker-compose.yml)

## 8.1 The big picture

The model you serve should not depend on whatever happens to be on someone's laptop.
Instead, **GitHub trains it, checks it, versions it, and packages it**:

```
push to main (model code)  ──►  "Train model" workflow
                                 test → train → accuracy ≥ 95%? → GitHub Release "model-vN"
                                   assets: mnist_model.npz, .sha256, metrics.json
                                        │ triggers
push to main (API code)    ──►  "Build image" workflow
                                 test → find latest model release
                                 → docker build (downloads + verifies model)
                                 → push ghcr.io/himmat12/numpy-digit-recognizer
                                        │
                     docker compose up  ──►  API on :8000 serving model-vN
```

**Why GitHub Releases for the model?** Release files are permanent, versioned
(`model-v1`, `model-v2`, ...) and downloadable without logging in. Plain workflow
"artifacts" are deleted after 90 days and need a token to download.

**Why bake the model into the image?** Each image is tied to exactly one model version, so
`ghcr.io/...:model-v3` always serves the same model. That makes deployments reproducible and
rollbacks trivial.

## 8.2 Workflow 1: `train-model.yml`

**When it runs:** on a push to `main` that changes the model code (`digit_recognizer/**` or
`requirements.txt`), or manually from GitHub → *Actions* → *Train model* → *Run workflow*.

**Steps:**

1. **Set up Python 3.11** and cache pip downloads between runs.
2. **Cache MNIST** (`actions/cache`), so it's downloaded once rather than on every run.
3. **Run the tests.** Never publish a model built from broken code.
4. **Train.** This is the same `python -m digit_recognizer.train` you run locally. It also
   writes `models/metrics.json` with the test accuracy and settings.
5. **Quality gate.** If test accuracy is below `MIN_ACCURACY` (95%), the run fails and nothing
   is published. This stops a bad change from shipping a bad model.
6. **Checksum.** `sha256sum` creates a fingerprint of the weights file.
7. **Publish.** `gh release create model-v<run number>` uploads the weights, checksum and
   metrics. The release notes show the accuracy and the commit it was trained from.

The job has `contents: write` permission. That's the minimum needed to create releases.

## 8.3 The Dockerfile

It has two **stages**. Only the second one becomes the final image.

**Stage 1 `model`: fetch and verify.**
```dockerfile
ARG MODEL_URL=https://github.com/himmat12/numpy-digit-recognizer/releases/latest/download
ADD ${MODEL_URL}/mnist_model.npz ${MODEL_URL}/mnist_model.npz.sha256 ./
RUN sha256sum -c mnist_model.npz.sha256 && chmod 644 mnist_model.npz
```
If the download is corrupted or tampered with, the checksum won't match and the build stops.
(`ADD` saves downloaded files readable only by root, hence the `chmod`.)

**Stage 2 `runtime`: the service.**
- Installs only `requirements-api.txt`. Copying the requirements files *before* the code
  means Docker reuses the cached install layer when only code changes.
- Copies just `digit_recognizer/`, `api/` and the verified model. `.dockerignore` keeps out
  `.venv`, `data/`, local `models/`, tests and docs.
- Runs as a normal user (`appuser`), not root, which limits the damage if the service is ever
  compromised.
- Has a `HEALTHCHECK` that calls `/health` every 30 seconds, so `docker ps` shows
  `healthy` / `unhealthy`.
- Records `MODEL_VERSION` so `/health` and every prediction report which model is serving.

## 8.4 Workflow 2: `build-image.yml`

**When it runs:**
- after *Train model* **succeeds**: a new model means a new image;
- on a push to `main` that changes `api/**`, the `Dockerfile` or `requirements-api.txt`:
  new code means a new image with the current model;
- manually.

**Steps:**

1. Run the tests.
2. **Find the latest model release** with `gh release view`. If none exists yet, it prints a
   warning and skips the build, instead of failing.
3. Log in to the GitHub Container Registry using the built-in `GITHUB_TOKEN`.
4. **Build and push** with the exact release URL (`MODEL_URL=.../download/model-vN`), so the
   image always contains the model the workflow just looked up. Tags:
   - `latest`: the newest build
   - `model-vN`: the model version inside
   - `sha-abc1234`: the code commit it was built from

Build layers are cached in GitHub Actions (`cache-from/to: type=gha`), which makes rebuilds
fast.

## 8.5 Running the service with Docker

**Use the published image** (no Python needed on your machine):

```bash
docker compose pull
docker compose up -d
curl localhost:8000/health        # {"status":"ok","model_version":"model-v1"}
```

Or run a specific model version:

```bash
docker run -p 8000:8000 ghcr.io/himmat12/numpy-digit-recognizer:model-v1
```

> **First time only:** new GHCR packages are private. To pull without `docker login`, open
> the package on GitHub (your profile → *Packages* → *numpy-digit-recognizer* →
> *Package settings*) and set its visibility to **Public**.

**Build locally** (downloads the latest model release from GitHub):

```bash
docker compose build
docker compose up -d
```

## 8.6 Common tasks

| I want to...                    | Do this                                                            |
|---------------------------------|--------------------------------------------------------------------|
| retrain with the current code   | Actions → *Train model* → *Run workflow*                           |
| change hyperparameters          | edit defaults in `digit_recognizer/train.py`, push to `main`       |
| see a model's accuracy          | open its release on GitHub, or download `metrics.json`             |
| roll back to an older model     | `docker run ... ghcr.io/himmat12/numpy-digit-recognizer:model-v1`  |
| change the accuracy threshold   | `MIN_ACCURACY` in `train-model.yml`                                |

## Ideas to extend

- Add a `pull_request` trigger that only runs the tests, so changes are checked before merging.
- Deploy the image automatically (e.g. to a cloud container service) after it's pushed.
- Have the API reject requests larger than a few MB.
