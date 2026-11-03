"""Team finances of an export save: the financial overview of the league, one row per MLB team, with the money as numbers.

The export has a `finance` folder in the season folder with one csv (ID, Team Name, Pay, Bgt, BgtSpc, ...). `BgtSpc` (budget
space) is the money still available for the current season. The teams are named by their full name there; the player files use
the team code, which comes from the `[<save>.team_names]` table of config/saves.toml.
"""

import re

import numpy as np
import pandas as pd

from util import paths
from util.config import get_save

MONEY_COLUMNS = {
    "Tkt": "Ticket", "Pay": "Payroll", "Bgt": "Budget", "BgtSpc": "BudgetSpace", "SctBgt": "ScoutBudget", "DevBgt": "DevBudget",
    "DrfBgt": "DraftBudget", "IAFABgt": "IAFABudget", "Revenue": "Revenue", "Exp": "Expenses", "CO": "CashOut",
}  # fmt: skip
TEXT_COLUMNS = {"Mkt": "Market", "Loy": "Loyalty", "Mode": "Mode"}
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9}


def parse_money(value) -> float:
    """Dollars from the export's text: "$201.9m", "$19 000 038", "$638k", "$0"; NaN for "-", empty or missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower().replace(",", "").replace(" ", " ")
    match = re.fullmatch(r"-?\$?\s*(-?[\d ]*\.?\d+)\s*([kmb])?", text)
    if not match:
        return np.nan
    number = float(match.group(1).replace(" ", ""))
    return number * _SCALE.get(match.group(2), 1.0)


def finance_files(save: str, season: int):
    folder = paths.finance_dir(save, season)
    return sorted(folder.glob("*.csv")) if folder.exists() else []


def read_finances(save: str, season: int) -> pd.DataFrame | None:
    """One row per team: Team (the code of the player files; NaN when config/saves.toml does not name the team), Team Name and
    the money columns as dollars (see MONEY_COLUMNS); None when the season has no finance folder."""
    files = finance_files(save, season)
    if not files:
        return None
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates()
    codes = {name: code for code, name in get_save(save).team_names.items()}
    out = pd.DataFrame({"Team": raw["Team Name"].map(codes), "Team Name": raw["Team Name"]})
    for source, name in MONEY_COLUMNS.items():
        if source in raw.columns:
            out[name] = raw[source].map(parse_money)
    for source, name in TEXT_COLUMNS.items():
        if source in raw.columns:
            out[name] = raw[source]
    return out


def write_finances(save: str, season: int) -> pd.DataFrame | None:
    """Write `<save>-<season>-finances.csv` next to the other outputs (nothing when the export has no finance folder)."""
    table = read_finances(save, season)
    if table is not None:
        table.to_csv(paths.output_file(save, season, "finances"), index=False)
    return table
