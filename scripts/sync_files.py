"""Store the league files in S3 and fetch them on another machine (the models have scripts/sync_models.py).

Usage:
    PYTHONPATH=. python scripts/sync_files.py push                      # the raw files of the human leagues
    PYTHONPATH=. python scripts/sync_files.py pull                      # on another machine
    PYTHONPATH=. python scripts/sync_files.py push --set viewer         # what the viewer server needs (after update_all.py)
    PYTHONPATH=. python scripts/sync_files.py pull --set viewer
    PYTHONPATH=. python scripts/sync_files.py push --set training       # the dumps of the training leagues (large)
    PYTHONPATH=. python scripts/sync_files.py list --set raw
    PYTHONPATH=. python scripts/sync_files.py push --dry-run

Sets (each has its own history of dated manifests, as in sync_models.py; files are stored once under blobs/<sha256>, shared with
the models):
  raw       everything the human leagues of config/update.toml have under files/<save>/ except the processed `output` folders
            (update_all.py makes those again): ratings and statistics exports, finance folders, the ballpark csv.
  viewer    what the viewer needs for the latest season of each human league: the processed files it reads and the ballpark csv.
  training  the dumps and other raw files of the dump leagues of config/saves.toml (Model-Sim ...), for retraining elsewhere.

A pull only downloads files that differ and never deletes. The bucket is `ootp-models` (OOTP_MODELS_BUCKET or --bucket); the
credentials are those of `aws configure`, and the bucket policy has to allow the machine.
"""

import argparse
from pathlib import Path

from scripts import sync_models
from util import paths
from util.config import PROJECT_ROOT, get_save
from util.update_config import load_update_config

FILES = PROJECT_ROOT / "files"
SETS = ("raw", "viewer", "training")
MANIFESTS = {name: f"files-{name}" for name in SETS}
VIEWER_OUTPUTS = (  # the processed files the viewer reads (viewer/queries.py, data_cache.py, parks.py)
    "outlook-slots", "outlook-players", "player-data", "predicted-values", "finances", "park-rates", "woba-calcs",
    "team-pitching", "growth",
)  # fmt: skip
SKIPPED_FOLDERS = ("output",)


def _relative(root: Path, paths_: list[Path]) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in sorted(paths_) if p.is_file()}


def raw_files(root: Path, saves: list[str]) -> dict[str, Path]:
    """Every file under the save folders except the processed `output` folders."""
    found = []
    for save in saves:
        base = root / save
        found += [
            p for p in base.rglob("*") if p.is_file() and not set(p.relative_to(base).parts[:-1]) & set(SKIPPED_FOLDERS)
        ]
    return _relative(root, found)


def latest_season(root: Path, save: str) -> int | None:
    """The newest season folder of a save that has processed outlook files."""
    seasons = [
        int(p.name)
        for p in (root / save).glob("[0-9]*")
        if p.name.isdigit() and (p / "output" / f"{save}-{p.name}-outlook-slots.csv").exists()
    ]
    return max(seasons) if seasons else None


def viewer_files(root: Path, saves: list[str], ballparks: dict[str, str | None]) -> dict[str, Path]:
    """The viewer's processed files of the latest season of each save and its ballpark csv (the season's own, else the shared)."""
    found = []
    for save in saves:
        season = latest_season(root, save)
        if season is None:
            continue
        output = root / save / str(season) / "output"
        found += [output / f"{save}-{season}-{name}.csv" for name in VIEWER_OUTPUTS]
        name = ballparks.get(save)
        if name:
            found += [p for p in (root / save / str(season) / Path(name).name, root / save / name) if p.exists()][:1]
    return _relative(root, found)


def collect(
    kind: str, root: Path = FILES, saves: list[str] | None = None, ballparks: dict | None = None
) -> dict[str, Path]:
    """The files of a set (relative path under `root` -> file). `saves`: the leagues of the set; by default those of the
    configuration (the human leagues of config/update.toml, or for `training` the dump leagues of config/saves.toml)."""
    if kind not in SETS:
        raise ValueError(f"unknown set {kind!r}; known: {SETS}")
    if saves is None:
        saves = list(load_update_config()) if kind != "training" else training_saves()
    if kind == "raw" or kind == "training":
        return raw_files(root, saves)
    ballparks = {save: get_save(save).ballparks for save in saves} if ballparks is None else ballparks
    return viewer_files(root, saves, ballparks)


def training_saves() -> list[str]:
    from util.config import load_saves

    return [name for name, save in load_saves().items() if save.source == "dump" and (paths.save_dir(name)).is_dir()]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Store the league files in S3 and fetch them on another machine.")
    parser.add_argument("action", choices=["push", "pull", "list"])
    parser.add_argument("--set", dest="kind", choices=SETS, default="raw", help="which files (default raw)")
    parser.add_argument("--version", help="pull: a set name (or its beginning) from `list`; default the latest")
    parser.add_argument("--bucket", default=sync_models.BUCKET)
    parser.add_argument("--dry-run", action="store_true", help="say what would be transferred")
    args = parser.parse_args(argv)

    import boto3  # here so that the helpers above can be tested without it

    client = boto3.client("s3")
    manifests = MANIFESTS[args.kind]
    if args.action == "push":
        files = collect(args.kind)
        print(f"set {args.kind}: {len(files)} files under {FILES}")
        sync_models.push(client, args.bucket, FILES, dry_run=args.dry_run, files=files, manifests=manifests)
    elif args.action == "pull":
        sync_models.pull(client, args.bucket, FILES, version=args.version, dry_run=args.dry_run, manifests=manifests)
    else:
        sync_models.show(client, args.bucket, manifests)


if __name__ == "__main__":
    main()
