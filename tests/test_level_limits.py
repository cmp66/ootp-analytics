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


def test_a_level_has_a_roster_maximum_and_a_pitcher_target_that_default_to_none(tmp_path):
    path = write(tmp_path, '[S.levels."AA"]\nmax_players = 26\npitchers = 13\n[S.levels."A"]\nmax_age = 25\n')
    levels = load_saves(path)["S"].levels
    assert (
        levels["AA"] == LevelLimits(max_players=26, pitchers=13)
        and levels["A"].max_players == 0
        and levels["A"].pitchers == 0
    )
    with pytest.raises(ValueError):
        load_saves(write(tmp_path, '[S.levels."AA"]\npitchers = -1\n'))
    path = write(tmp_path, '[S.levels."INT"]\nmax_players = 50\nno_demotion = true\n')
    assert load_saves(path)["S"].levels["INT"] == LevelLimits(max_players=50, no_demotion=True)
    assert LevelLimits().no_demotion is False


def test_bad_level_limits_are_refused(tmp_path):
    with pytest.raises(ValueError):
        load_saves(write(tmp_path, '[S.levels."AA"]\nmax_year = 3\n'))
    with pytest.raises(ValueError):
        load_saves(write(tmp_path, '[S.levels."AA"]\nmax_age = -1\n'))


def test_the_shipped_config_lists_every_level_of_the_human_leagues():
    for name in ("TGS27",):
        levels = load_saves()[name].levels
        assert set(levels) >= {"AAA", "AA", "A+", "A-", "R+", "R", "INT"}
        assert all(isinstance(limit, LevelLimits) for limit in levels.values())


def test_a_level_can_be_kept_off_the_promotion_ladder_and_the_table_order_is_kept(tmp_path):
    path = write(
        tmp_path, '[S.levels."AAA"]\nmax_age = 30\n[S.levels."WL"]\nladder = false\n[S.levels."AA"]\nmax_age = 27\n'
    )
    levels = load_saves(path)["S"].levels
    assert list(levels) == ["AAA", "WL", "AA"] and levels["WL"].ladder is False and levels["AAA"].ladder is True
    assert levels["WL"] == LevelLimits(ladder=False)
