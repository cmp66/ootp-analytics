import numpy as np
import pandas as pd
import pytest

from model import artifact, registry
from scripts import derived_ratings, rating_importance
from tests.test_model_registry import multi_season_root


def test_the_derived_rating_fit_recovers_known_weights():
    rng = np.random.default_rng(0)
    first = rng.integers(4, 17, 5000) * 5.0
    second = rng.integers(4, 17, 5000) * 5.0
    derived = np.round((0.8 * first + 0.2 * second) / 5) * 5  # an exact 80 / 20 rule shown in steps of 5
    result = derived_ratings.fit(pd.DataFrame({"D": derived, "A": first, "B": second}), "D", "A", "B")
    assert result["share_first"] == pytest.approx(0.8, abs=0.02)
    assert result["first"] == pytest.approx(0.8, abs=0.03) and result["second"] == pytest.approx(0.2, abs=0.03)
    assert result["r2"] > 0.95 and result["resid_sd"] < 2.0 and result["n"] == 5000


def test_zero_ratings_mean_not_applicable_and_are_left_out():
    frame = pd.DataFrame({"D": [50, 60, 0, 70], "A": [50, 60, 40, 70], "B": [50, 60, 40, 0]})
    assert derived_ratings.fit(frame, "D", "A", "B")["n"] == 2


def test_the_ranking_table_has_one_row_per_rating_in_order():
    table = pd.DataFrame(
        {
            "feature": ["POW", "K"],
            "r2_drop": [0.4, 0.2],
            "effect_step": [9.6, 6.6],
            "effect_sd": [9.0, 7.0],
            "sd": [10.0, 11.0],
            "share": [2 / 3, 1 / 3],
        }
    )
    lines = rating_importance.format_table(table)
    assert len(lines) == 4 and lines[2].startswith("| 1 | Power (`POW`) | 67% |") and "+9.60 (+10 points)" in lines[2]
    assert lines[3].startswith("| 2 | Avoid strikeouts (AVK) (`K`) | 33% |")
    flipped = rating_importance.format_table(table, divisor=-0.5)
    assert "-19.20" in flipped[2]  # the divisor turns a BABIP difference into runs saved


@pytest.mark.integration
def test_importance_ranks_the_ratings_of_a_trained_model(tmp_path, monkeypatch):
    multi_season_root(tmp_path, monkeypatch)
    spec = registry.specs(["baserunning"])[0]
    loaded = registry.train_spec(spec, "Model-Sim", "Standard27", [2046, 2047, 2048, 2049], cv=False)
    ranking = rating_importance.importance_for(loaded, "Model-Sim", folds=3, repeats=2)
    assert set(ranking["feature"]) == {"SPE", "STE", "RUN"}
    assert ranking["r2_drop"].is_monotonic_decreasing and ranking["r2_drop"].iloc[0] > 0
    assert ranking["share"].sum() == pytest.approx(1.0)
    assert (ranking["sd"] > 0).all()
    assert isinstance(loaded, artifact.Artifact)
