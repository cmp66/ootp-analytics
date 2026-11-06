import gzip

import numpy as np
import pandas as pd
import pytest

from model import growth


def league(values, potential=None, pitcher=False):
    """A snapshot of players with every rating of a kind set to the player's value (and the potential to his potential)."""
    ids = range(1, len(values) + 1)
    cols = growth.PITCHER if pitcher else growth.HITTER
    pots = growth.PITCHER_POTENTIAL if pitcher else growth.HITTER_POTENTIAL
    frame = pd.DataFrame({c: values for c in cols}, index=pd.Index(ids, name="ID"))
    for c in pots:
        frame[c] = potential if potential is not None else values
    frame["POS"] = "SP" if pitcher else "SS"
    return frame


def test_a_player_who_gains_while_the_league_stands_still_grows_and_the_rest_do_not():
    old = league([40, 45, 50, 55, 60] * 4)
    new = old.copy()
    new.loc[1, growth.HITTER] = 50  # +10 rating points on a league spread of about 7
    result = growth.growth(old, new, 365).set_index("ID")
    assert result.loc[1, "d_cur"] > 3 and result.loc[1, "d_pot"] == pytest.approx(0, abs=1e-9)
    # his gain moves the mean of this small league, which is taken off: the others slip a little
    assert result.drop(1)["d_cur"].abs().max() < 1.0


def test_a_shift_of_the_whole_league_is_not_growth():
    old = league([40, 45, 50, 55, 60] * 4)
    new = old.copy()
    new[growth.HITTER] = old[growth.HITTER] + 5  # a rebalance
    result = growth.growth(old, new, 365)
    assert result["d_cur"].abs().max() < 1e-9


def test_a_shorter_span_is_scaled_to_a_year_and_the_change_is_capped():
    old = league([40, 45, 50, 55, 60] * 4)
    new = old.copy()
    new.loc[1, growth.HITTER] += 1
    year, half = growth.growth(old, new, 365).set_index("ID"), growth.growth(old, new, 182.5).set_index("ID")
    assert half.loc[1, "d_cur"] == pytest.approx(2 * year.loc[1, "d_cur"], rel=0.05)
    wild = old.copy()
    wild.loc[1, growth.HITTER] = 200
    assert growth.growth(old, wild, 365).set_index("ID").loc[1, "d_cur"] == growth.CAP


def test_pitchers_are_measured_on_their_own_ratings():
    old = league([40, 45, 50, 55, 60] * 4, pitcher=True)
    new = old.copy()
    new.loc[2, growth.PITCHER_POTENTIAL] += 8
    result = growth.growth(old, new, 365).set_index("ID")
    assert result.loc[2, "d_pot"] > 2 and result.loc[2, "d_cur"] == pytest.approx(0, abs=1e-9)


def test_players_in_only_one_snapshot_get_no_growth_and_zero_ratings_are_not_ratings():
    old = league([40, 45, 50, 55, 60] * 4)
    new = old.drop(index=[3]).copy()
    new.loc[1, growth.HITTER] = 0  # no rating
    result = growth.growth(old, new, 365).set_index("ID")
    assert 3 not in result.index and np.isnan(result.loc[1, "d_cur"])


def write_snapshot(folder, frame):
    folder.mkdir(parents=True)
    with gzip.open(folder / "players.csv.gz", "wt") as out:
        frame.reset_index().to_csv(out, index=False)


def test_the_archive_gives_growth_only_with_two_exports_far_enough_apart(monkeypatch, tmp_path):
    monkeypatch.setattr(growth.paths, "archive_dir", lambda save, season, date: tmp_path / str(season) / date)
    old = league([40, 45, 50, 55, 60] * 4)
    new = old.copy()
    new.loc[1, growth.HITTER] = 50
    assert growth.archive_growth("X", 2046) is None  # nothing archived
    write_snapshot(tmp_path / "2046" / "2046-02-26", old)
    assert growth.archive_growth("X", 2046) is None  # one export
    write_snapshot(tmp_path / "2046" / "2046-03-05", new)
    assert growth.archive_growth("X", 2046) is None  # a week apart: too little
    write_snapshot(tmp_path / "2046" / "2046-07-01", new)
    result = growth.archive_growth("X", 2046)
    assert result is not None and result["days"].iloc[0] == 125 and result.set_index("ID").loc[1, "d_cur"] > 0
