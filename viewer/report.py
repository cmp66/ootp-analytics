"""Reports of the viewer: an ordered list, a block per position, and the filters that produced them (plan 15).

Pure functions over the outlook frame of `viewer.queries`, so the same tables go to the screen and to an Excel workbook.
"""

from dataclasses import dataclass, field
from io import BytesIO

import pandas as pd

from viewer import queries


@dataclass
class ReportSpec:
    """What the report shows; `filters` are (label, text) lines printed at the top of every sheet."""

    title: str
    metric: str
    horizon: int
    filters: list[tuple[str, str]] = field(default_factory=list)
    top: int = 50
    top_per_position: int = 25


def ordered_list(pool: pd.DataFrame, spec: ReportSpec, lowest: bool = False, with_id: bool = False) -> pd.DataFrame:
    """Each player once, at his best position (the chosen ones when the filters name positions); `lowest` reverses the order."""
    if not lowest:
        rows = queries.tier_list(pool, spec.metric, spec.horizon, spec.top, by_player=True)
    else:
        ranked = pool.assign(key=queries.metric_column(spec.metric, spec.horizon, pool).to_numpy())
        ranked = ranked.sort_values(["key", f"rank_score_h{spec.horizon}"])
        rows = ranked.drop_duplicates("ID").head(spec.top).reset_index(drop=True)
    return queries.display_table(rows, spec.horizon, with_id)


def position_blocks(pool: pd.DataFrame, spec: ReportSpec, with_id: bool = False) -> dict[str, pd.DataFrame]:
    """One table per position that has players, in the usual order: a player appears at every position he qualifies for."""
    blocks = {}
    for slot in queries.SLOT_ORDER:
        part = pool[pool["slot"] == slot]
        if part.empty:
            continue
        rows = queries.tier_list(part, spec.metric, spec.horizon, spec.top_per_position)
        blocks[slot] = queries.display_table(rows, spec.horizon, with_id).drop(columns=["slot"], errors="ignore")
    return blocks


def write_excel(
    ordered: pd.DataFrame, lowest: pd.DataFrame, blocks: dict[str, pd.DataFrame], spec: ReportSpec
) -> bytes:
    """Workbook with the sheets Ordered list, Lowest, By position; each starts with the title and the filters."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        book = writer.book
        bold = book.add_format({"bold": True})
        note = book.add_format({"italic": True, "font_color": "#555555"})
        header = book.add_format({"bold": True, "bottom": 1})

        def preface(sheet) -> int:
            sheet.write(0, 0, spec.title, bold)
            for i, (label, text) in enumerate(spec.filters, start=1):
                sheet.write(i, 0, f"{label}: {text}", note)
            return len(spec.filters) + 2

        def put(sheet, frame: pd.DataFrame, row: int, col: int) -> int:
            frame = frame.drop(columns=["ID"], errors="ignore")  # the player ID is for the screens
            for j, name in enumerate(frame.columns):
                sheet.write(row, col + j, name, header)
                width = max(
                    len(str(name)), *(len(f"{v:.2f}" if isinstance(v, float) else str(v)) for v in frame[name].head(60))
                )
                sheet.set_column(col + j, col + j, min(width + 2, 22))
            for i, values in enumerate(frame.itertuples(index=False), start=row + 1):
                for j, v in enumerate(values):
                    if pd.isna(v):
                        continue
                    sheet.write(i, col + j, round(v, 3) if isinstance(v, float) else v)
            return col + len(frame.columns)

        for name, frame in (("Ordered list", ordered), ("Lowest", lowest)):
            sheet = book.add_worksheet(name)
            start = preface(sheet)
            put(sheet, frame, start, 0)
            sheet.freeze_panes(start + 1, 3)
        sheet = book.add_worksheet("By position")
        start = preface(sheet)
        col = 0
        for slot, frame in blocks.items():
            sheet.write(start, col, slot, bold)
            col = put(sheet, frame, start + 1, col) + 1
    return buffer.getvalue()
