"""Service settings, read from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from digit_recognizer.train import MODEL_PATH as DEFAULT_MODEL_PATH


@dataclass(frozen=True)
class Settings:
    model_path: Path
    model_version: str


def get_settings() -> Settings:
    """Build settings from the environment.

    MODEL_PATH     where the trained weights (.npz) live
    MODEL_VERSION  label of the model, e.g. the GitHub release tag "model-v3"
    """
    return Settings(
        model_path=Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)),
        model_version=os.environ.get("MODEL_VERSION", "local"),
    )
