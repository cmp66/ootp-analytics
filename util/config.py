import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "saves.toml"

SOURCES = ("dump", "export")


@dataclass(frozen=True)
class SaveConfig:
    name: str
    source: str
    ratings: str
    # ballpark ratings csv, relative to files/<save>/ (a season folder may hold its own copy)
    ballparks: str | None = None
    # share of a player's plate appearances / batters faced that happen in his own park
    home_share: float = 0.5
    # save-specific team identifier -> "Team Name" in the ballpark csv (export: ratings TM abbreviation)
    team_names: dict[str, str] = field(default_factory=dict)


def load_saves(path: Path = CONFIG_PATH) -> dict[str, SaveConfig]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    saves = {}
    for name, values in raw.items():
        if values.get("source") not in SOURCES:
            raise ValueError(f"Save {name}: source must be one of {SOURCES}, got {values.get('source')!r}")
        home_share = values.get("home_share", 0.5)
        if not 0 <= home_share <= 1:
            raise ValueError(f"Save {name}: home_share must be between 0 and 1, got {home_share}")
        saves[name] = SaveConfig(
            name=name,
            source=values["source"],
            ratings=values["ratings"],
            ballparks=values.get("ballparks"),
            home_share=home_share,
            team_names=dict(values.get("team_names", {})),
        )
    return saves


def get_save(name: str) -> SaveConfig:
    saves = load_saves()
    if name not in saves:
        raise KeyError(f"Unknown save {name!r}; add it to {CONFIG_PATH} (known: {sorted(saves)})")
    return saves[name]
