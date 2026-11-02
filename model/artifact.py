"""One file per trained model: the fitted model (weights and scaler), the feature names and order, what it predicts
and how it was trained. Loading needs no notebook and no knowledge of the training code beyond this module.

Layout: files/models/<save>/<ratings>/<name>.joblib. The file is a pickle (joblib); only load files you made."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from util import paths

FORMAT_VERSION = 1


@dataclass
class Artifact:
    name: str
    model: object
    features: list[str]
    target: str
    save: str
    ratings: str
    family: str
    seasons: list[int]
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)  # for example the platoon mix of a split model
    commit: str | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predictions for rows of ratings; columns are matched by name, so their order does not matter."""
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise KeyError(f"{self.name}: missing feature columns {missing}")
        return self.model.predict(X[self.features])


def models_dir(save: str, ratings: str) -> Path:
    return paths.FILES_ROOT / "models" / save / ratings


def artifact_path(save: str, ratings: str, name: str) -> Path:
    return models_dir(save, ratings) / f"{name}.joblib"


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except OSError, subprocess.CalledProcessError:
        return None


def save_artifact(artifact: Artifact) -> Path:
    path = artifact_path(artifact.save, artifact.ratings, artifact.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if artifact.commit is None:
        artifact.commit = git_commit()
    joblib.dump({"format": FORMAT_VERSION, "artifact": artifact}, path)
    return path


def load_artifact(save: str, ratings: str, name: str) -> Artifact:
    path = artifact_path(save, ratings, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; train it with scripts/train_models.py")
    payload = joblib.load(path)
    if payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported artifact format {payload.get('format')!r}")
    return payload["artifact"]
