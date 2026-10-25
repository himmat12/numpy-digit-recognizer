# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: download the trained model from a GitHub Release and verify it.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS model

# Folder that holds the release assets. Defaults to the latest model release;
# CI passes an exact version, e.g. .../releases/download/model-v3
ARG MODEL_URL=https://github.com/himmat12/numpy-digit-recognizer/releases/latest/download

WORKDIR /model
ADD ${MODEL_URL}/mnist_model.npz ${MODEL_URL}/mnist_model.npz.sha256 ./
# Fail the build if the download is corrupted or does not match its checksum.
# ADD saves remote files as root-only (0600), so also make the model world-readable.
RUN sha256sum -c mnist_model.npz.sha256 && chmod 644 mnist_model.npz

# ---------------------------------------------------------------------------
# Stage 2: the API service (only code, dependencies and the verified model).
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ARG MODEL_VERSION=latest

LABEL org.opencontainers.image.title="numpy-digit-recognizer" \
      org.opencontainers.image.description="FastAPI service for a NumPy handwritten digit recognizer" \
      org.opencontainers.image.source="https://github.com/himmat12/numpy-digit-recognizer"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/mnist_model.npz \
    MODEL_VERSION=${MODEL_VERSION}

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY digit_recognizer/ digit_recognizer/
COPY api/ api/
COPY --from=model /model/mnist_model.npz models/mnist_model.npz

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
