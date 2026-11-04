import pytest

from scripts import sync_files, sync_models
from tests.test_sync_models import FakeS3
from util import seasons


def put(root, relative, content=b"x"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture
def files(tmp_path):
    root = tmp_path / "files"
    for name in sync_files.VIEWER_OUTPUTS + ("hitting", "lists-extra"):
        put(root, f"L1/2046/output/L1-2046-{name}.csv", name.encode())
        put(root, f"L1/2045/output/L1-2045-{name}.csv")  # an older season
    put(root, "L1/2046/ratings/players.csv", b"ratings")
    put(root, "L1/2046/finance/finances.csv", b"money")
    put(root, "L1/ballpark_ratings.csv", b"parks")
    put(root, "L2/2082/output/L2-2082-outlook-slots.csv")  # another league with no other files
    put(root, "Train/dumps/Std/2045/players.csv", b"dump")
    return root


def test_the_raw_set_has_every_file_but_the_processed_output(files):
    found = sync_files.collect("raw", files, ["L1"])
    assert set(found) == {"L1/2046/ratings/players.csv", "L1/2046/finance/finances.csv", "L1/ballpark_ratings.csv"}
    assert set(sync_files.collect("training", files, ["Train"])) == {"Train/dumps/Std/2045/players.csv"}


def test_the_viewer_set_has_the_files_the_viewer_reads_of_the_latest_season_and_the_ballparks(files):
    found = sync_files.collect("viewer", files, ["L1", "L2"], {"L1": "ballpark_ratings.csv", "L2": None})
    assert {p for p in found if p.startswith("L1/2046/output")} == {
        f"L1/2046/output/L1-2046-{name}.csv" for name in sync_files.VIEWER_OUTPUTS
    }
    assert "L1/ballpark_ratings.csv" in found and not any("2045" in p or "ratings/players" in p for p in found)
    assert "L2/2082/output/L2-2082-outlook-slots.csv" in found
    with pytest.raises(ValueError):
        sync_files.collect("nothing", files, [])


def test_a_set_is_stored_under_its_own_history_and_comes_back_whole(files, tmp_path, monkeypatch):
    s3 = FakeS3()
    monkeypatch.setattr(sync_models, "git_commit", lambda: "abc1234")
    raw = sync_files.collect("raw", files, ["L1"])
    viewer = sync_files.collect("viewer", files, ["L1"], {"L1": "ballpark_ratings.csv"})
    first = sync_models.push(s3, "b", files, files=raw, manifests=sync_files.MANIFESTS["raw"])
    sync_models.push(s3, "b", files, files=viewer, manifests=sync_files.MANIFESTS["viewer"])
    assert sync_models.manifest_names(s3, "b", "files-raw") == [first]
    assert len(sync_models.manifest_names(s3, "b", "files-viewer")) == 1
    other = tmp_path / "other"
    sync_models.pull(s3, "b", other, manifests="files-viewer")
    assert (other / "L1" / "2046" / "output" / "L1-2046-outlook-slots.csv").read_bytes() == b"outlook-slots"
    assert not (other / "L1" / "2046" / "ratings").exists()  # the viewer set has no raw files


def test_the_viewer_can_show_a_season_that_only_has_its_processed_files(files, monkeypatch):
    from util import paths

    monkeypatch.setattr(paths, "FILES_ROOT", files)
    monkeypatch.setattr(seasons, "get_save", lambda name: type("S", (), {"source": "export"})())
    assert seasons.served_seasons("L1") == [2045, 2046]  # the two with outlook files; 2046 also has ratings
    assert seasons.available_seasons("L2") == []  # the processing needs the raw files
    assert seasons.served_seasons("L2") == [2082]
