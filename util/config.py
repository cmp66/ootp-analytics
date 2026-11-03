import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "saves.toml"

SOURCES = ("dump", "export")


@dataclass(frozen=True)
class LevelLimits:
    """The limits of a minor league level (0 = no limit): a player older than `max_age` or with more than `max_pro_years` years of
    professional experience does not belong at the level. The order of the level tables of a save is its promotion ladder, top
    level first."""

    max_age: int = 0
    max_pro_years: int = 0
    ladder: bool = True  # False: not a rung of the promotion ladder (the minor league management page leaves it alone)


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
    # level of the ratings column Lev (AAA, AA, A+, A-, R+, R, INT, ...) -> its limits; a level that is not listed has none
    levels: dict[str, LevelLimits] = field(default_factory=dict)
    # name of the rating scale profile of a league whose ratings are not all 20-80 (the models for it are trained on ratings
    # turned to that mix, see model/scales.py); None: all 20-80
    scales: str | None = None


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
        levels = {}
        for level, limits in values.get("levels", {}).items():
            unknown = set(limits) - {"max_age", "max_pro_years", "ladder"}
            if unknown:
                raise ValueError(f"Save {name}, level {level}: unknown keys {sorted(unknown)}")
            levels[level] = LevelLimits(
                **{key: int(value) for key, value in limits.items() if key != "ladder"}, ladder=bool(limits.get("ladder", True))
            )
            if min(levels[level].max_age, levels[level].max_pro_years) < 0:
                raise ValueError(f"Save {name}, level {level}: limits cannot be negative (0 = no limit)")
        saves[name] = SaveConfig(
            name=name,
            source=values["source"],
            ratings=values["ratings"],
            ballparks=values.get("ballparks"),
            home_share=home_share,
            team_names=dict(values.get("team_names", {})),
            levels=levels,
            scales=values.get("scales"),
        )
    return saves


def get_save(name: str) -> SaveConfig:
    saves = load_saves()
    if name not in saves:
        raise KeyError(f"Unknown save {name!r}; add it to {CONFIG_PATH} (known: {sorted(saves)})")
    return saves[name]
