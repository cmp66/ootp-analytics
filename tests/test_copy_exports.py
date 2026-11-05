import os

import pytest

from scripts import copy_exports
from util import paths

GOOD = "ID,B,T,Age\n1,L,R,30\n"
NAMES = ("a_person.csv", "b_person.csv")
FINANCE = "fin.csv"


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """files/S/7/{ratings,finance} holding old copies (time 1000) and an export folder with new ones (time 2000)."""
    monkeypatch.setattr(paths, "FILES_ROOT", tmp_path / "files")
    monkeypatch.setattr(copy_exports, "available_seasons", lambda save: [6, 7])
    ratings, finance = paths.export_ratings_dir("S", 7), paths.finance_dir("S", 7)
    exports = tmp_path / "exports" / "S"
    for folder in (ratings, finance, exports):
        folder.mkdir(parents=True)
    for name in (*NAMES, FINANCE):
        target = (finance if name == FINANCE else ratings) / name
        target.write_text("old")
        os.utime(target, (1000, 1000))
        (exports / name).write_text(GOOD)
        os.utime(exports / name, (2000, 2000))
    (exports / "extra_person.csv").write_text(GOOD)  # not in the season folder: ignored
    return exports, ratings, finance, tmp_path / "exports"


def test_all_new_files_are_copied_and_a_second_run_finds_nothing_new(layout):
    _, ratings, finance, root = layout
    plan = copy_exports.run(["S"], exports=root)[0]
    assert not plan.problems and len(plan.copy) == 3
    assert (ratings / NAMES[0]).read_text() == GOOD and (finance / FINANCE).read_text() == GOOD
    assert not (ratings / "extra_person.csv").exists()
    again = copy_exports.plan_save("S", root)
    assert again.copy == [] and "ratings: nothing new" in again.notes


def test_a_mix_of_new_and_old_ratings_files_copies_nothing(layout):
    exports, ratings, finance, root = layout
    os.utime(exports / NAMES[1], (500, 500))
    plan = copy_exports.run(["S"], exports=root)[0]
    assert plan.problems and plan.copy == []
    assert (ratings / NAMES[0]).read_text() == "old" and (finance / FINANCE).read_text() == "old"


def test_a_missing_ratings_file_is_a_problem(layout):
    exports, _, _, root = layout
    (exports / NAMES[1]).unlink()
    plan = copy_exports.plan_save("S", root)
    assert plan.problems and plan.copy == []


def test_a_repeated_b_column_is_a_problem(layout):
    exports, _, _, root = layout
    (exports / NAMES[0]).write_text("ID,B,B,T\n1,Left,L,R\n")
    os.utime(exports / NAMES[0], (2000, 2000))
    plan = copy_exports.plan_save("S", root)
    assert any("B appears 2 times" in p for p in plan.problems) and plan.copy == []


def test_a_finance_file_that_is_not_new_is_skipped_without_a_problem(layout):
    exports, _, finance, root = layout
    os.utime(exports / FINANCE, (500, 500))
    plan = copy_exports.plan_save("S", root)
    assert not plan.problems and len(plan.copy) == 2
    assert any("finance: not new" in n for n in plan.notes)


def test_ratings_that_are_all_old_copy_nothing_but_new_finance_still_copies(layout):
    exports, _, _, root = layout
    for name in NAMES:
        os.utime(exports / name, (500, 500))
    plan = copy_exports.plan_save("S", root)
    assert not plan.problems and [t.name for _, t in plan.copy] == [FINANCE]


def test_dry_run_copies_nothing(layout):
    _, ratings, _, root = layout
    copy_exports.run(["S"], dry_run=True, exports=root)
    assert (ratings / NAMES[0]).read_text() == "old"
