"""The saves of config/update.toml (the human leagues the update script refreshes and the viewer shows)."""

import tomllib
from pathlib import Path

UPDATE_CONFIG = Path(__file__).resolve().parent.parent / "config" / "update.toml"


def load_update_config(path: Path = UPDATE_CONFIG) -> dict[str, dict]:
    with open(path, "rb") as f:
        return tomllib.load(f)
