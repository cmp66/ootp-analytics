"""Store the trained models in S3 and fetch them on another machine.

Usage:
    PYTHONPATH=. python scripts/sync_models.py push                  # upload the models that changed, add a manifest
    PYTHONPATH=. python scripts/sync_models.py pull                  # download the latest set
    PYTHONPATH=. python scripts/sync_models.py pull --version 20261103T141500Z-ab12cd3   # an earlier set
    PYTHONPATH=. python scripts/sync_models.py list                  # the sets in the bucket (the history)
    PYTHONPATH=. python scripts/sync_models.py push --dry-run

Every .joblib under files/models/<save>/ (value and outlook models) is synced; the legacy .pt files are not.
The bucket is `ootp-models` (OOTP_MODELS_BUCKET overrides it) and the credentials are the ones of `aws configure`. Each file is
stored once under blobs/<sha256> and every push writes manifests/<time>-<git commit>.json, which maps the paths to the blobs,
so unchanged files are never uploaded twice and every earlier set can be restored. Nothing is ever deleted from the bucket.
"""

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

MODELS = Path("files/models")
BUCKET = os.environ.get("OOTP_MODELS_BUCKET", "ootp-models")
WORKERS = 8


def local_files(root: Path = MODELS) -> dict[str, Path]:
    """Relative posix path -> file for every .joblib model in the save folders of the models folder."""
    return {
        p.relative_to(root).as_posix(): p
        for d in sorted(root.iterdir())
        if d.is_dir()
        for p in sorted(d.rglob("*.joblib"))
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "nogit"


def manifest_names(client, bucket: str, manifests: str = "manifests") -> list[str]:
    """Names of the stored sets, oldest first (the names start with the UTC time). `manifests`: the key prefix of the set kind."""
    names = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{manifests}/"):
        names += [item["Key"].removeprefix(f"{manifests}/").removesuffix(".json") for item in page.get("Contents", [])]
    return sorted(names)


def stored_blobs(client, bucket: str) -> set[str]:
    blobs = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="blobs/"):
        blobs |= {item["Key"].removeprefix("blobs/") for item in page.get("Contents", [])}
    return blobs


def read_manifest(client, bucket: str, name: str, manifests: str = "manifests") -> dict:
    body = client.get_object(Bucket=bucket, Key=f"{manifests}/{name}.json")["Body"].read()
    return json.loads(body)


def push(
    client,
    bucket: str,
    root: Path = MODELS,
    dry_run: bool = False,
    files: dict[str, Path] | None = None,
    manifests: str = "manifests",
) -> str | None:
    """Upload the files that are not in the bucket and write a manifest of the set. `files` (relative path -> file) defaults to
    the models under `root`; `manifests` is the key prefix of this kind of set."""
    files = local_files(root) if files is None else files
    if not files:
        raise SystemExit(f"no files to store under {root}")
    with ThreadPoolExecutor(WORKERS) as pool:
        hashes = dict(zip(files, pool.map(sha256, files.values())))
    stored = stored_blobs(client, bucket)
    new = {path: digest for path, digest in hashes.items() if digest not in stored}
    # the same content in two paths is uploaded once
    unique = {digest: path for path, digest in new.items()}
    size = sum(files[path].stat().st_size for path in unique.values())
    print(f"{len(files)} files, {len(new)} not in the bucket ({len(unique)} different, {size / 1e6:.1f} MB)")
    if dry_run:
        return None
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda item: client.upload_file(str(files[item[1]]), bucket, f"blobs/{item[0]}"), unique.items()))
    name = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{git_commit()}"
    taken = manifest_names(client, bucket, manifests)
    suffix = 1
    while name in taken:  # two pushes in the same second
        suffix += 1
        name = f"{name.split('+')[0]}+{suffix}"
    manifest = {
        "name": name,
        "files": {path: {"sha256": hashes[path], "size": files[path].stat().st_size} for path in files},
    }
    client.put_object(Bucket=bucket, Key=f"{manifests}/{name}.json", Body=json.dumps(manifest, indent=1).encode())
    print(f"stored set {name}")
    return name


def resolve(names: list[str], version: str | None) -> str:
    if not names:
        raise SystemExit("the bucket has no stored sets; run push first")
    if version is None:
        return names[-1]
    found = [n for n in names if n == version] or [n for n in names if n.startswith(version)]
    if len(found) != 1:
        raise SystemExit(f"version {version!r} matches {len(found)} sets; `list` shows them")
    return found[0]


def pull(
    client,
    bucket: str,
    root: Path = MODELS,
    version: str | None = None,
    dry_run: bool = False,
    manifests: str = "manifests",
) -> str:
    name = resolve(manifest_names(client, bucket, manifests), version)
    files = read_manifest(client, bucket, name, manifests)["files"]
    todo = {}
    for path, info in files.items():
        target = root / path
        if not (target.is_file() and target.stat().st_size == info["size"] and sha256(target) == info["sha256"]):
            todo[path] = info
    print(
        f"set {name}: {len(files)} files, {len(todo)} to download ({sum(i['size'] for i in todo.values()) / 1e6:.1f} MB)"
    )
    if dry_run:
        return name

    def fetch(item):
        path, info = item
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        client.download_file(bucket, f"blobs/{info['sha256']}", str(part))
        os.replace(part, target)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(fetch, todo.items()))
    return name


def show(client, bucket: str, manifests: str = "manifests") -> None:
    for name in reversed(manifest_names(client, bucket, manifests)):
        manifest = read_manifest(client, bucket, name, manifests)
        total = sum(info["size"] for info in manifest["files"].values())
        print(f"{name}  {len(manifest['files'])} files  {total / 1e6:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Store the trained models in S3 and fetch them on another machine.")
    parser.add_argument("action", choices=["push", "pull", "list"])
    parser.add_argument("--version", help="pull: a set name (or its beginning) from `list`; default the latest")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--dry-run", action="store_true", help="say what would be transferred")
    args = parser.parse_args()

    import boto3  # here so that the helpers above can be tested without it

    client = boto3.client("s3")
    if args.action == "push":
        push(client, args.bucket, dry_run=args.dry_run)
    elif args.action == "pull":
        pull(client, args.bucket, version=args.version, dry_run=args.dry_run)
    else:
        show(client, args.bucket)


if __name__ == "__main__":
    main()
