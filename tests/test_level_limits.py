import pytest

from util.config import LevelLimits, load_saves


def write(tmp_path, body):
    path = tmp_path / "saves.toml"
    path.write_text('[S]\nsource = "export"\nratings = "R"\n' + body)
    return path


def test_level_limits_default_to_no_limit_and_are_read_per_level(tmp_path):
    path = write(tmp_path, '[S.levels."A+"]\nmax_age = 24\n[S.levels."AA"]\nmax_age = 0\nmax_pro_years = 5\n')
    levels = load_saves(path)["S"].levels
    assert levels["A+"] == LevelLimits(max_age=24, max_pro_years=0)
    assert levels["AA"] == LevelLimits(max_age=0, max_pro_years=5)
    assert load_saves(write(tmp_path, ""))["S"].levels == {}


def test_bad_level_limits_are_refused(tmp_path):
    with pytest.raises(ValueError):
        load_saves(write(tmp_path, '[S.levels."AA"]\nmax_year = 3\n'))
    with pytest.raises(ValueError):
        load_saves(write(tmp_path, '[S.levels."AA"]\nmax_age = -1\n'))


def test_the_shipped_config_lists_every_level_of_the_human_leagues_with_no_limit():
    for name in ("TGS", "TGS26"):
        levels = load_saves()[name].levels
        assert set(levels) >= {"AAA", "AA", "A+", "A-", "R+", "R", "INT"}
        assert all(limit == LevelLimits() for limit in levels.values())
