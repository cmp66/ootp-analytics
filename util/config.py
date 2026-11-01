import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "saves.toml"

SOURCES = ("dump", "export")


@dataclass(frozen=True)
class SaveConfig:
    name: str
    source: str
    ratings: str


def load_saves(path: Path = CONFIG_PATH) -> dict[str, SaveConfig]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    saves = {}
    for name, values in raw.items():
        if values.get("source") not in SOURCES:
            raise ValueError(
                f"Save {name}: source must be one of {SOURCES}, got {values.get('source')!r}"
            )
        saves[name] = SaveConfig(
            name=name, source=values["source"], ratings=values["ratings"]
        )
    return saves


def get_save(name: str) -> SaveConfig:
    saves = load_saves()
    if name not in saves:
        raise KeyError(
            f"Unknown save {name!r}; add it to {CONFIG_PATH} (known: {sorted(saves)})"
        )
    return saves[name]
