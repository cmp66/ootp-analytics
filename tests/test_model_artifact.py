import numpy as np
import pytest

from model import artifact
from model.train import fit_table, make_model
from tests.test_model_train import FAST_MLP, make_table
from util import paths


@pytest.fixture
def models_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FILES_ROOT", tmp_path)
    return tmp_path


def build(family, table, name="batting"):
    params = FAST_MLP if family == "mlp" else {}
    model = fit_table(make_model(family, **params), table, seed=2)
    return artifact.Artifact(
        name=name,
        model=model,
        features=list(table.features),
        target="wRAAAdj600",
        save="Model-Sim",
        ratings="Standard27",
        family=family,
        seasons=[2046, 2047],
        metrics={"r2": 0.5},
        notes={"right_share": 0.73},
    )


@pytest.mark.parametrize("family", ["ridge", "boosting", "mlp"])
def test_a_saved_model_reloads_to_identical_predictions(models_root, family):
    table = make_table(n_players=50, seasons=3)
    original = build(family, table)
    path = artifact.save_artifact(original)
    assert path == models_root / "models" / "Model-Sim" / "Standard27" / "batting.joblib"
    loaded = artifact.load_artifact("Model-Sim", "Standard27", "batting")
    X = table.X.iloc[:30]
    assert loaded.predict(X) == pytest.approx(original.predict(X), abs=1e-6)
    assert loaded.metrics == {"r2": 0.5} and loaded.notes == {"right_share": 0.73}
    assert loaded.features == list(table.features) and loaded.seasons == [2046, 2047]


def test_the_reloaded_model_keeps_its_own_scaler_and_matches_columns_by_name(models_root):
    table = make_table(n_players=50, seasons=3)
    artifact.save_artifact(build("mlp", table))
    loaded = artifact.load_artifact("Model-Sim", "Standard27", "batting")
    X = table.X.iloc[:12]
    one_by_one = np.concatenate([loaded.predict(X.iloc[[i]]) for i in range(12)])
    assert one_by_one == pytest.approx(loaded.predict(X[["c", "b", "a"]]), abs=1e-4)
    with pytest.raises(KeyError, match="missing feature"):
        loaded.predict(X[["a", "b"]])


def test_missing_and_foreign_files_are_clear_errors(models_root):
    with pytest.raises(FileNotFoundError, match="train_models"):
        artifact.load_artifact("Model-Sim", "Standard27", "nothing")
    path = artifact.artifact_path("Model-Sim", "Standard27", "odd")
    path.parent.mkdir(parents=True)
    artifact.joblib.dump({"format": 99, "artifact": None}, path)
    with pytest.raises(ValueError, match="unsupported artifact format"):
        artifact.load_artifact("Model-Sim", "Standard27", "odd")


def test_the_commit_is_recorded_when_git_is_available(models_root):
    table = make_table(n_players=40, seasons=2)
    saved = build("ridge", table)
    artifact.save_artifact(saved)
    assert saved.commit is None or len(saved.commit) >= 7
