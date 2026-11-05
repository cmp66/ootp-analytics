"""Set up another machine to develop and run the project: the libraries, the models and the league files from S3, the processed
files, and a check that it all works.

Usage (after `git clone`, with Python 3.14 and `uv` installed and `aws configure` done):
    python scripts/setup_machine.py --install cpu            # libraries (cpu, cuda or xpu), models, raw files, update
    python scripts/setup_machine.py                          # the same without installing libraries (they are already there)
    python scripts/setup_machine.py --test                   # also run the test suite at the end
    python scripts/setup_machine.py --training               # also the dumps of the training leagues (16 GB), to retrain here
    python scripts/setup_machine.py --dry-run                # say what it would do

Steps, in order (each one that fails stops the run and says what to fix):
  install   the requirements file of the hardware (only with --install): `uv pip install` as in the README
  check     the libraries import (pandas, scikit-learn, torch, streamlit, boto3 ...)
  aws       the credentials work and the bucket answers (`aws configure`; the bucket policy must allow this machine's IP)
  models    sync_models.py pull: the latest trained models into files/models/
  files     sync_files.py pull: the raw files of the human leagues (ratings and statistics exports, finance, ballparks)
  training  sync_files.py pull --set training (only with --training)
  update    update_all.py: the processed files, values, outlook scores and lists, so the viewer can start
  test      python -m pytest (only with --test)
Everything it pulls is only downloaded when it differs and nothing is deleted. Then: python scripts/run_viewer.py
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import sync_models  # noqa: E402

MODULES = ("numpy", "pandas", "scipy", "sklearn", "torch", "joblib", "streamlit", "boto3", "xlsxwriter")
PYTORCH = "https://download.pytorch.org/whl"
INSTALL = {  # the commands of the README, Setup
    "cpu": f"uv pip install -r requirements-cpu.txt --index-url {PYTORCH}/cpu --extra-index-url https://pypi.org/simple "
    "--index-strategy unsafe-best-match",
    "cuda": "uv pip install -r requirements-cuda.txt",
    "xpu": f"uv pip install -r requirements-xpu.txt --index-url {PYTORCH}/xpu --extra-index-url https://pypi.org/simple "
    "--index-strategy unsafe-best-match",
}  # fmt: skip


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...] | None  # None: a check done in this process
    what: str


def script(name: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, str(ROOT / "scripts" / name), *arguments)


def steps(install: str | None = None, training: bool = False, update: bool = True, test: bool = False) -> list[Step]:
    """The steps to run, in order (see the module docstring)."""
    found = []
    if install:
        found.append(Step("install", tuple(INSTALL[install].split()), f"the libraries for {install}"))
    found += [
        Step("check", None, "the libraries import"),
        Step("aws", None, "the credentials work and the bucket answers"),
        Step("models", script("sync_models.py", "pull"), "the trained models into files/models/"),
        Step("files", script("sync_files.py", "pull"), "the raw files of the human leagues"),
    ]
    if training:
        found.append(
            Step("training", script("sync_files.py", "pull", "--set", "training"), "the dumps of the training leagues")
        )
    if update:
        found.append(Step("update", script("update_all.py"), "the processed files, values, outlook scores and lists"))
    if test:
        found.append(Step("test", (sys.executable, "-m", "pytest", "-q"), "the test suite"))
    return found


def missing_modules(modules=MODULES) -> list[str]:
    """The libraries that cannot be imported."""
    return [name for name in modules if importlib.util.find_spec(name) is None]


def check_environment() -> str:
    """ "" when the libraries are there, else what to do."""
    if sys.version_info < (3, 14):
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        return f"Python 3.14 is needed, this is {version} (pyenv: the ootp-analytics virtualenv)"
    missing = missing_modules()
    if missing:
        return f"missing libraries: {', '.join(missing)}. Run with --install cpu (or cuda, xpu), see the README, Setup"
    return ""


def check_aws(bucket: str) -> str:
    """ "" when the credentials work and the bucket can be listed, else what to do."""
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        boto3.client("s3").list_objects_v2(Bucket=bucket, Prefix="manifests/", MaxKeys=1)
    except ImportError:
        return "boto3 is not installed: run with --install"
    except (BotoCoreError, ClientError) as error:
        return (
            f"cannot reach the bucket {bucket}: {error}. Run `aws configure` with the credentials that may use it, and check "
            "that "
            "the bucket policy allows the IP address of this machine"
        )
    return ""


def run(selected: list[Step], bucket: str, dry_run: bool = False) -> int:
    """Run the steps; the first that fails stops the run (returns 1)."""
    for number, step in enumerate(selected, 1):
        print(f"[{number}/{len(selected)}] {step.name}: {step.what}", flush=True)
        if dry_run:
            print("    " + (" ".join(step.command) if step.command else "(check)"))
            continue
        if step.command is None:
            problem = check_environment() if step.name == "check" else check_aws(bucket)
            if problem:
                print(f"    FAILED: {problem}", file=sys.stderr)
                return 1
            continue
        done = subprocess.run(step.command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
        if done.returncode:
            print(
                f"    FAILED ({step.name}, exit status {done.returncode}); fix it and run the script again",
                file=sys.stderr,
            )
            return 1
    print("Done. Start the viewer with: python scripts/run_viewer.py" if not dry_run else "Dry run: nothing was done.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Set up this machine to develop and run the project.")
    parser.add_argument("--install", choices=sorted(INSTALL), help="install the libraries first (the hardware)")
    parser.add_argument("--training", action="store_true", help="also pull the dumps of the training leagues (16 GB)")
    parser.add_argument("--no-update", action="store_true", help="do not run update_all.py at the end")
    parser.add_argument("--test", action="store_true", help="run the test suite at the end")
    parser.add_argument("--bucket", default=sync_models.BUCKET)
    parser.add_argument("--dry-run", action="store_true", help="say what would be done")
    args = parser.parse_args(argv)
    (ROOT / "files").mkdir(exist_ok=True)
    return run(steps(args.install, args.training, not args.no_update, args.test), args.bucket, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
