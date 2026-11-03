import pytest

from scripts import sync_models


class FakeS3:
    """The few calls of sync_models, on a dict of key -> bytes."""

    def __init__(self):
        self.objects = {}

    def get_paginator(self, name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": key} for key in sorted(objects) if key.startswith(Prefix)]}

        return Paginator()

    def upload_file(self, filename, bucket, key):
        with open(filename, "rb") as handle:
            self.objects[key] = handle.read()

    def download_file(self, bucket, key, filename):
        with open(filename, "wb") as handle:
            handle.write(self.objects[key])

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        data = self.objects[Key]

        class Body:
            def read(self):
                return data

        return {"Body": Body()}


@pytest.fixture
def models(tmp_path):
    root = tmp_path / "models"
    (root / "Save" / "R" / "outlook").mkdir(parents=True)
    (root / "Save" / "R" / "a.joblib").write_bytes(b"aaa")
    (root / "Save" / "R" / "outlook" / "b.joblib").write_bytes(b"bbbb")
    (root / "legacy.pt").write_bytes(b"old")
    return root


def test_push_uploads_only_what_changed(models, monkeypatch):
    s3 = FakeS3()
    monkeypatch.setattr(sync_models, "git_commit", lambda: "abc1234")
    first = sync_models.push(s3, "b", models)
    assert sorted(k for k in s3.objects if k.startswith("blobs/")).__len__() == 2
    (models / "Save" / "R" / "a.joblib").write_bytes(b"changed")
    second = sync_models.push(s3, "b", models)
    assert len([k for k in s3.objects if k.startswith("blobs/")]) == 3
    assert sync_models.manifest_names(s3, "b") == sorted([first, second])
    assert "legacy.pt" not in sync_models.read_manifest(s3, "b", first)["files"]


def test_pull_restores_a_version_and_skips_unchanged(models, tmp_path, monkeypatch):
    s3 = FakeS3()
    monkeypatch.setattr(sync_models, "git_commit", lambda: "abc1234")
    first = sync_models.push(s3, "b", models)
    (models / "Save" / "R" / "a.joblib").write_bytes(b"changed")
    sync_models.push(s3, "b", models)

    target = tmp_path / "other"
    (target / "Save").mkdir(parents=True)
    sync_models.pull(s3, "b", target)
    assert (target / "Save" / "R" / "a.joblib").read_bytes() == b"changed"
    assert (target / "Save" / "R" / "outlook" / "b.joblib").read_bytes() == b"bbbb"

    sync_models.pull(s3, "b", target, version=first)
    assert (target / "Save" / "R" / "a.joblib").read_bytes() == b"aaa"
    assert not list(target.rglob("*.part"))


def test_pull_without_sets_stops(tmp_path):
    with pytest.raises(SystemExit):
        sync_models.pull(FakeS3(), "b", tmp_path)


def test_dry_run_changes_nothing(models, tmp_path):
    s3 = FakeS3()
    assert sync_models.push(s3, "b", models, dry_run=True) is None
    assert s3.objects == {}
