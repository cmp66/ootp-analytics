import json

from scripts import update_all
from viewer import queries


def fake_config(monkeypatch, tmp_path, calls, fail=None):
    monkeypatch.setattr(update_all, "load_update_config", lambda: {"A": {"models_from": "M"}, "B": {}})
    monkeypatch.setattr(update_all, "get_save", lambda name: object())
    monkeypatch.setattr(update_all, "available_seasons", lambda save: [2045, 2046])
    monkeypatch.setattr(update_all, "input_signature", lambda save, season: {"f": [1, 2]})
    monkeypatch.setattr(update_all, "stamp_file", lambda save, season: tmp_path / f"{save}-{season}.json")

    def make(name):
        def run(job):
            calls.append((name, job.save, job.season, job.models_from))
            return 1 if (name, job.save) == fail else 0

        return update_all.Step(name, name, run)

    monkeypatch.setattr(update_all, "STEPS", [make("one"), make("two")])


def test_runs_all_steps_of_all_saves_on_latest_season(monkeypatch, tmp_path):
    calls = []
    fake_config(monkeypatch, tmp_path, calls)
    assert update_all.main([]) == 0
    assert calls == [("one", "A", 2046, "M"), ("two", "A", 2046, "M"), ("one", "B", 2046, "B"), ("two", "B", 2046, "B")]


def test_failed_step_stops_that_save_only(monkeypatch, tmp_path):
    calls = []
    fake_config(monkeypatch, tmp_path, calls, fail=("one", "A"))
    assert update_all.main([]) == 1
    assert [c[:2] for c in calls] == [("one", "A"), ("one", "B"), ("two", "B")]
    assert not (tmp_path / "A-2046.json").exists() and (tmp_path / "B-2046.json").exists()


def test_if_changed_skips_unchanged_inputs(monkeypatch, tmp_path):
    calls = []
    fake_config(monkeypatch, tmp_path, calls)
    update_all.main([])
    calls.clear()
    assert json.loads((tmp_path / "A-2046.json").read_text()) == {"f": [1, 2]}
    update_all.main(["--if-changed"])
    assert calls == []
    monkeypatch.setattr(update_all, "input_signature", lambda save, season: {"f": [1, 3]})
    update_all.main(["A", "--if-changed", "--only", "two"])
    assert calls == [("two", "A", 2046, "M")]


def test_unknown_step_or_save_is_an_error(monkeypatch, tmp_path):
    fake_config(monkeypatch, tmp_path, [])
    assert update_all.main(["--only", "nope"]) == 2
    assert update_all.main(["C"]) == 2


def test_real_steps_cover_the_scripts():
    assert [s.name for s in update_all.STEPS] == ["process", "values", "park", "outlook", "lists"]
    assert "TGS27" in update_all.load_update_config()


def test_pool_flags_need_the_scouting_column_loaded():
    assert "SctCat" in queries.IDENTITY_COLUMNS
