import os
from pathlib import Path

import pytest

from scripts import sync_models, sync_notes
from tests.test_sync_models import FakeS3


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A repository with plans/ and a Claude Code project folder with a memory note and a conversation."""
    monkeypatch.setattr(sync_models, "git_commit", lambda: "abc1234")
    root = tmp_path / "repo"
    (root / "plans" / "baseline").mkdir(parents=True)
    (root / "plans" / "00-overview.md").write_text("overview")
    (root / "plans" / "baseline" / "out.csv").write_text("a,b")
    (root / "plans" / "__pycache__").mkdir()
    (root / "plans" / "__pycache__" / "x.pyc").write_bytes(b"x")
    project = tmp_path / "claude" / "projects" / "repo"
    (project / "memory").mkdir(parents=True)
    (project / "memory" / "MEMORY.md").write_text("- note")
    (project / "session1.jsonl").write_text("{}")
    (project / "session1").mkdir()
    (project / "session1" / "result.txt").write_text("out")
    return root, project


def test_the_claude_folder_of_a_project_is_its_path_with_other_characters_replaced():
    folder = sync_notes.claude_project_dir(Path("/home/cmp11/projects/ootp-analytics"), Path("/h"))
    assert folder == Path("/h/.claude/projects/-home-cmp11-projects-ootp-analytics")
    assert (
        sync_notes.claude_project_dir(Path("C:/Users/me/ootp analytics"), Path("/h")).name
        == "C--Users-me-ootp-analytics"
    )


def test_each_set_has_its_files_and_the_conversations_leave_the_memory_notes_to_their_own_set(machine):
    root, project = machine
    assert set(sync_notes.collect("memory", sync_notes.set_root("memory", root, project))) == {"MEMORY.md"}
    assert set(sync_notes.collect("plans", sync_notes.set_root("plans", root, project))) == {
        "00-overview.md",
        "baseline/out.csv",
    }
    assert set(sync_notes.collect("sessions", sync_notes.set_root("sessions", root, project))) == {
        "session1.jsonl",
        "session1/result.txt",
    }
    assert sync_notes.collect("plans", root / "missing") == {}


def test_a_push_stores_each_set_apart_and_a_pull_on_another_machine_restores_it(machine, tmp_path, capsys):
    root, project = machine
    s3 = FakeS3()
    sync_notes.push(s3, "b", ("memory", "plans"), root, project)
    assert {k.split("/")[0] for k in s3.objects} == {
        "notes-memory",
        "notes-plans",
        "notes-blobs",
    }  # not under blobs/ or manifests/
    other = tmp_path / "other"
    (other / "plans").mkdir(parents=True)
    other_project = tmp_path / "other-claude"
    sync_notes.pull(s3, "b", ("memory", "plans"), other, other_project)
    assert (other_project / "memory" / "MEMORY.md").read_text() == "- note"
    assert (other / "plans" / "baseline" / "out.csv").read_text() == "a,b" and not (
        other / "plans" / "__pycache__"
    ).exists()
    sync_notes.push(s3, "b", ("memory",), root, project, dry_run=True)  # a dry run stores nothing new
    assert len(sync_models.manifest_names(s3, "b", "notes-memory")) == 1


def test_a_pull_leaves_alone_a_file_changed_after_the_push_unless_forced(machine, tmp_path, capsys):
    root, project = machine
    s3 = FakeS3()
    sync_notes.push(s3, "b", ("memory",), root, project)
    note = project / "memory" / "MEMORY.md"
    note.write_text("- edited here")
    future = note.stat().st_mtime + 3600
    os.utime(note, (future, future))  # changed after the set was pushed
    sync_notes.pull(s3, "b", ("memory",), root, project)
    assert note.read_text() == "- edited here" and "left alone: MEMORY.md" in capsys.readouterr().out
    sync_notes.pull(s3, "b", ("memory",), root, project, force=True)
    assert note.read_text() == "- note"
    old = (note.stat().st_mtime - 86400 * 2,) * 2
    note.write_text("- from an old edit")
    os.utime(note, old)  # older than the set: the stored one replaces it
    sync_notes.pull(s3, "b", ("memory",), root, project)
    assert note.read_text() == "- note"


def test_mirror_removes_old_local_files_that_are_not_in_the_set_and_keeps_new_ones(machine, capsys):
    root, project = machine
    s3 = FakeS3()
    sync_notes.push(s3, "b", ("memory",), root, project)
    memory = project / "memory"
    old, new = memory / "old.md", memory / "new.md"
    old.write_text("x")
    new.write_text("y")
    ancient = old.stat().st_mtime - 86400 * 2
    os.utime(old, (ancient, ancient))
    future = new.stat().st_mtime + 3600
    os.utime(new, (future, future))
    sync_notes.pull(s3, "b", ("memory",), root, project, mirror=True, dry_run=True)
    assert old.exists() and "would remove: old.md" in capsys.readouterr().out  # a dry run removes nothing
    sync_notes.pull(s3, "b", ("memory",), root, project, mirror=True)
    assert not old.exists() and new.exists() and "kept: new.md" in capsys.readouterr().out


def test_the_time_of_a_set_comes_from_its_name():
    assert sync_notes.set_time("20261105T170355Z-1f4ca5d") == 1793898235.0
