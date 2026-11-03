"""Start the viewer in the browser (http://localhost:8501).

Usage:
    PYTHONPATH=. python scripts/run_viewer.py [--port 8501]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "viewer" / "app.py"),
        "--server.port",
        str(args.port),
    ]
    return subprocess.call(command, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})


if __name__ == "__main__":
    sys.exit(main())
