import pandas as pd
import pytest

from model import artifact
from scripts import evaluate_models, predict_values, train_models
from stats import pipeline
from tests.test_model_registry import multi_season_root
from util import paths


@pytest.fixture
def root(tmp_path, monkeypatch):
    return multi_season_root(tmp_path, monkeypatch)


@pytest.mark.integration
def test_train_predict_and_evaluate_from_the_scripts(root, capsys):
    assert (
        train_models.main(["Model-Sim", "--start", "2046", "--stop", "2050", "--streams", "batting", "baserunning"])
        == 0
    )
    out = capsys.readouterr().out
    assert "Training 3 models on Model-Sim 2046-2050" in out and "batting-split (ridge)" in out and "cv R2" in out
    folder = artifact.models_dir("Model-Sim", "Standard27")
    assert sorted(p.stem for p in folder.glob("*.joblib")) == ["baserunning", "batting", "batting-split"]
    saved = artifact.load_artifact("Model-Sim", "Standard27", "batting-split")
    assert saved.seasons == [2046, 2047, 2048, 2049, 2050]

    assert predict_values.main(["Model-Sim", "--season", "2051"]) == 0
    result = pd.read_csv(paths.output_file("Model-Sim", 2051, "predicted-values"))
    assert {"ID", "batting", "batting_vs_right", "batting_vs_left", "batting_plain", "baserunning"} <= set(
        result.columns
    )
    assert result["batting"].notna().sum() > 100 and result["ID"].is_unique
    capsys.readouterr()

    assert evaluate_models.main(["Model-Sim", "--start", "2051", "--stop", "2051", "--calibration"]) == 0
    text = capsys.readouterr().out
    assert "unseen" in text and "total value (blend of the two sides)" in text and "calibration" in text


@pytest.mark.integration
def test_scripts_default_to_all_seasons_for_training_and_the_latest_for_prediction(root, capsys):
    assert train_models.main(["Model-Sim", "--no-cv", "--no-plain", "--streams", "baserunning"]) == 0
    assert (
        "Model-Sim 2045-2051" in capsys.readouterr().out
    )  # the dump folders of the fixture: 2045 and the copies 2046-2051
    assert predict_values.main(["Model-Sim"]) == 0
    assert paths.output_file("Model-Sim", 2051, "predicted-values").exists()


@pytest.mark.integration
def test_models_of_one_save_score_another(root, capsys):
    assert (
        train_models.main(["Model-Sim", "--start", "2046", "--stop", "2050", "--no-cv", "--streams", "baserunning"])
        == 0
    )
    pipeline.process_season("TGS", 2046)
    assert predict_values.main(["TGS", "--models-from", "Model-Sim"]) == 0
    result = pd.read_csv(paths.output_file("TGS", 2046, "predicted-values"))
    assert result["baserunning"].notna().sum() > 100
    assert "(Model-Sim) ->" in capsys.readouterr().out


def test_bad_requests_are_clear_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "FILES_ROOT", tmp_path)
    assert train_models.main(["Nope"]) == 2
    assert predict_values.main(["Model-Sim"]) == 2
    assert evaluate_models.main(["Model-Sim"]) == 2
    with pytest.raises(SystemExit):
        train_models.main(["Model-Sim", "--streams", "bogus"])
    assert "error:" in capsys.readouterr().err
