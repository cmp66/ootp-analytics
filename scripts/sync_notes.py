"""Keep the things that are not in git in step between your machines: Claude Code's memory notes, the plans/ folder and (when you
ask) the Claude Code conversations of this project. Same design as sync_models.py and sync_files.py.

Usage:
    python scripts/sync_notes.py push                 # memory notes and plans/ to the bucket (when you finish some work)
    python scripts/sync_notes.py pull                 # on the other machine (before you start)
    python scripts/sync_notes.py push --set sessions  # the conversations too (so `claude --resume` can continue them there)
    python scripts/sync_notes.py push --all           # memory, plans and sessions
    python scripts/sync_notes.py pull --dry-run       # say what would change
    python scripts/sync_notes.py list                 # the stored sets, newest first
    python scripts/sync_notes.py pull --mirror        # also remove local files that are not in the set (older than it)
    python scripts/sync_notes.py pull --force         # also overwrite local files that are newer than the set

Sets:
  memory    Claude Code's memory notes of this project: ~/.claude/projects/<this folder>/memory/ (the folder name is the
            project path with every character that is not a letter or digit replaced by "-", so another path gives another name)
  plans     the plans/ folder of the repository (history of the larger pieces of work; not in git)
  sessions  the conversations of this project (the .jsonl files and their folders next to memory/); opt-in because they are big
            and hold everything that was on screen (paths, IP addresses, outputs). They are written to the folder of this machine.

A pull never deletes (unless --mirror) and leaves alone a local file that differs from the stored one and is NEWER than the set
(you changed it here after the last push): it says so, and --force overwrites it. So push before you leave a machine, pull before
you work on the next. Stored in the bucket of sync_models.py under notes-<set>/ (manifests) and notes-blobs/ (files), apart from
the models and league files: the read-only key of the viewer server cannot read them.
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import sync_models  # noqa: E402

SETS = ("memory", "plans", "sessions")
DEFAULT_SETS = ("memory", "plans")
BLOBS = "notes-blobs"
SKIPPED = ("__pycache__", ".DS_Store", ".part")


def manifests(kind: str) -> str:
    return f"notes-{kind}"


def claude_project_dir(root: Path = ROOT, home: Path | None = None) -> Path:
    """Where Claude Code keeps what it knows of the project at `root`: ~/.claude/projects/<the path with every character that is
    not a letter or digit replaced by "-">."""
    return (home or Path.home()) / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(root))


def set_root(kind: str, root: Path = ROOT, project_dir: Path | None = None) -> Path:
    """The local folder of a set; the files of the set are named relative to it."""
    project_dir = project_dir or claude_project_dir(root)
    return {"memory": project_dir / "memory", "plans": root / "plans", "sessions": project_dir}[kind]


def collect(kind: str, base: Path) -> dict[str, Path]:
    """The files of a set under `base` (relative path -> file). The conversations are everything of the project folder except
    the memory notes (their own set)."""
    if not base.is_dir():
        return {}
    found = {}
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base)
        if not path.is_file() or any(part in SKIPPED or part.endswith(".part") for part in relative.parts):
            continue
        if kind == "sessions" and relative.parts[0] == "memory":
            continue
        found[relative.as_posix()] = path
    return found


def set_time(name: str) -> float:
    """When a stored set was pushed (seconds since the epoch), from its name (it starts with the UTC time)."""
    return datetime.strptime(name[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


def newer_than(path: Path, when: float) -> bool:
    return path.stat().st_mtime > when


def extra_files(local: dict[str, Path], stored: dict, when: float) -> tuple[list[str], list[str]]:
    """The local files that are not in the stored set: (old enough to remove with --mirror, newer than the set: kept)."""
    extra = [path for path in local if path not in stored]
    return [p for p in extra if not newer_than(local[p], when)], [p for p in extra if newer_than(local[p], when)]


def push(client, bucket: str, kinds, root: Path = ROOT, project_dir: Path | None = None, dry_run: bool = False) -> None:
    for kind in kinds:
        base = set_root(kind, root, project_dir)
        files = collect(kind, base)
        print(f"== {kind}: {base}")
        if not files:
            print("   nothing there; skipped")
            continue
        sync_models.push(client, bucket, base, dry_run, files, manifests(kind), BLOBS)


def pull(
    client,
    bucket: str,
    kinds,
    root: Path = ROOT,
    project_dir: Path | None = None,
    version: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    mirror: bool = False,
) -> None:
    for kind in kinds:
        base = set_root(kind, root, project_dir)
        names = sync_models.manifest_names(client, bucket, manifests(kind))
        print(f"== {kind}: {base}")
        if not names:
            print("   nothing stored yet; skipped")
            continue
        name = sync_models.resolve(names, version)
        when = set_time(name)

        def protect(path, target, when=when):
            return (
                None
                if force or not newer_than(target, when)
                else "changed here after the set was pushed; --force overwrites it"
            )

        sync_models.pull(client, bucket, base, name, dry_run, manifests(kind), BLOBS, protect)
        if mirror:
            stored = sync_models.read_manifest(client, bucket, name, manifests(kind))["files"]
            removable, kept = extra_files(collect(kind, base), stored, when)
            for path in removable:
                print(f"{'would remove' if dry_run else 'removed'}: {path} (not in the set)")
                if not dry_run:
                    (base / path).unlink()
            for path in kept:
                print(f"kept: {path} (not in the set, but newer than it)")


def show(client, bucket: str, kinds) -> None:
    for kind in kinds:
        print(f"== {kind}")
        sync_models.show(client, bucket, manifests(kind))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Keep memory notes, plans and conversations in step between machines.")
    parser.add_argument("action", choices=["push", "pull", "list"])
    parser.add_argument(
        "--set", dest="kinds", nargs="+", choices=SETS, help=f"which sets (default {' '.join(DEFAULT_SETS)})"
    )
    parser.add_argument("--all", action="store_true", help="every set, the conversations too")
    parser.add_argument("--version", help="pull: a set name (or its beginning) from `list`; default the latest")
    parser.add_argument("--force", action="store_true", help="pull: overwrite local files that are newer than the set")
    parser.add_argument("--mirror", action="store_true", help="pull: remove local files that are not in the set")
    parser.add_argument(
        "--claude-dir", type=Path, help="the project folder of Claude Code (default from this folder's path)"
    )
    parser.add_argument("--bucket", default=sync_models.BUCKET)
    parser.add_argument("--dry-run", action="store_true", help="say what would be transferred")
    args = parser.parse_args(argv)
    kinds = SETS if args.all else tuple(args.kinds or DEFAULT_SETS)

    import boto3  # here so that the helpers above can be tested without it

    client = boto3.client("s3")
    if args.action == "push":
        push(client, args.bucket, kinds, project_dir=args.claude_dir, dry_run=args.dry_run)
    elif args.action == "pull":
        pull(client, args.bucket, kinds, project_dir=args.claude_dir, version=args.version, dry_run=args.dry_run,
             force=args.force, mirror=args.mirror)  # fmt: skip
    else:
        show(client, args.bucket, kinds)


if __name__ == "__main__":
    main()
