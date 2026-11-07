"""Player ratings page in the style of the game's player pages (plan 15): a pure function from a row of player-data to html.

Ratings are on the 20-80 scale of the export, or on the 1-10 scale in leagues that show them that way (a value from 1 to 10 is a
1-10 rating: no 20-80 rating is below 20); the number is shown as it is in the file, the color and the bar use the 20-80 value it
stands for. A bar shows the current rating, a lighter bar behind it the potential.
Hitters: basic batting ratings, running, summary, position and defensive ratings. Pitchers: basic pitching ratings, pitches.
Both: personality and injury proneness.
"""

import html

import pandas as pd

from viewer import injuries

CSS = """
<style>
.pc {background: #1b1c1f; color: #dcdcdc; border-radius: 8px; padding: 12px 16px; font-family: inherit;
     display: grid; grid-template-columns: 1.7fr 1fr; gap: 6px 28px; font-size: 15px;}
.pc h5 {margin: 10px 0 4px; font-size: 15px; font-weight: 700; color: #b9bcc2; text-transform: uppercase; letter-spacing: .02em;}
.pc h5:first-child {margin-top: 0;}
.pc .sub {color: #b9bcc2; font-weight: 600; margin: 0 0 2px 150px;}
.pc .row {display: grid; grid-template-columns: var(--label, 150px) var(--value, 78px) 1fr; align-items: center;
     gap: 0 6px; height: 26px;}
.pc .row.in {padding-left: 28px; grid-template-columns: calc(var(--label, 150px) - 28px) var(--value, 78px) 1fr;}
.pc .lab {color: #b9bcc2; font-weight: 600;}
.pc .val {text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; white-space: nowrap;}
.pc .val i {font-style: normal; color: #b9bcc2; padding: 0 3px;}
.pc .track {background: #3a3b3f; height: 14px; border-radius: 7px; position: relative; overflow: hidden;}
.pc .pot, .pc .cur {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px;}
.pc .pot {opacity: .7;}
.pc .two {display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px; --label: 105px; --value: 40px;}
.pc .two.wide {--label: 90px; --value: 74px;}
.pc .kv {display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px; margin-top: 6px;}
.pc .kv .text {justify-content: flex-start; gap: 14px;}
.pc .kv .lab {min-width: 92px;}
.pc .text {display: flex; justify-content: space-between; height: 24px; align-items: center;}
.pc .text b {color: #e8e8e8; font-weight: 600;}
.pc .big {display: flex; justify-content: space-between; height: 26px; align-items: center;}
.pc .note {color: #b9bcc2; margin: 2px 0;}
.pc .hi {color: #3dd10f;} .pc .lo {color: #fcc22b;}
.pc .name {grid-column: 1 / -1; font-size: 13px; color: #9a9da4; margin-bottom: 2px;}
</style>
"""
HITTER_BASIC = [
    ("Contact", "CON", "CON P", False), ("Avoid K's", "K's", "K P", True), ("BABIP", "BABIP", "BA P", True),
    ("Gap Power", "GAP", "GAP P", False), ("Power", "POW", "POW P", False), ("Eye", "EYE", "EYE P", False),
]  # fmt: skip
POTENTIAL_SHOWN = 40  # a position with no rating now is listed when his potential there is at least this
RUNNING_LEFT = [("Speed", "SPE"), ("Baserunning", "RUN"), ("Sacrifice Bunt", "BUN")]
RUNNING_RIGHT = [("Steal Tendency", "SR"), ("Stealing Ability", "STE"), ("Bunt for Hit", "BFH")]
POSITIONS = [
    ("C", "Catcher"), ("1B", "First Base"), ("2B", "Second Base"), ("3B", "Third Base"), ("SS", "Shortstop"),
    ("LF", "Left Field"), ("CF", "Center Field"), ("RF", "Right Field"),
]  # fmt: skip
DEFENSE = {
    "C": [("Catcher Ability", "C ABI"), ("Catcher Framing", "C FRM"), ("Catcher Arm", "C ARM")],
    "IF": [("Infield Range", "IF RNG"), ("Infield Error", "IF ERR"), ("Infield Arm", "IF ARM"), ("Turn Double Play", "TDP")],
    "OF": [("Outfield Range", "OF RNG"), ("Outfield Error", "OF ERR"), ("Outfield Arm", "OF ARM")],
}  # fmt: skip
PITCHER_BASIC = [
    ("Stuff", "STU", "STU P", False), ("Movement", "MOV", "MOV P", False), ("HR Allowed", "HRR", "HRR P", True),
    ("PBABIP", "PBABIP", "PBABIP P", True), ("Control", "CON.1", "CON.1 P", False),
]  # fmt: skip
GROUND_FLY = {62: "Extreme Ground Ball", 60: "Ground Ball", 50: "Neutral", 46: "Fly Ball", 42: "Extreme Fly Ball"}
PITCHER_TYPE = {10: "Ground Ball Pitcher", 20: "Normal Pitcher", 30: "Finesse Pitcher", 40: "Power Pitcher"}
ARM_SLOT = {4: "Over the Top", 3: "Three Quarters", 2: "Sidearm", 1: "Submarine"}
VELOCITY = {  # the export's text, as coded by model.convert_velocity
    2: "80-83", 3: "83-85", 4: "84-86", 5: "85-87", 6: "86-88", 7: "87-89", 8: "88-90", 9: "89-91", 10: "90-92",
    11: "91-93", 12: "92-94", 13: "93-95", 14: "94-96", 15: "95-97", 16: "96-98", 17: "97-99", 18: "98-100",
    19: "99-101", 20: "100+",
}  # fmt: skip
PITCHES = {
    "FB": "Fastball", "CH": "Changeup", "CB": "Curveball", "SL": "Slider", "SI": "Sinker", "SP": "Splitter",
    "CT": "Cutter", "FO": "Forkball", "CC": "Circle Change", "SC": "Screwball", "KC": "Knuckle Curve", "KN": "Knuckleball",
}  # fmt: skip
TRAITS = [("WE cat", "Work Ethic"), ("INT cat", "Intelligence"), ("LEA cat", "Leadership"), ("LOY cat", "Loyalty"),
          ("AD cat", "Adaptability"), ("FIN cat", "Greed")]  # fmt: skip


def as_20_80(value: float) -> float:
    """The 20-80 value a rating stands for: 1-10 ratings are spread evenly over 20 to 80; 0 (none) and 20-80 ratings stay."""
    return 20 + (value - 1) / 9 * 60 if 0 < value <= 10 else value


def color(value: float) -> str:
    """Rating color: red, orange, amber, green, teal, blue from low to high."""
    value = as_20_80(value)
    if value <= 30:
        return "#e8553d"
    if value <= 35:
        return "#f0813a"
    if value <= 45:
        return "#fcc22b"
    if value <= 55:
        return "#3dd10f"
    if value <= 65:
        return "#25b8a6"
    return "#0b9cf0"


def _number(row: pd.Series, column: str | None):
    if column is None or column not in row.index or pd.isna(row[column]):
        return None
    return int(row[column])


def _width(value: int) -> float:
    return max(3.0, min(100.0, (as_20_80(value) - 20) / 60 * 96))


def rating_row(label: str, now, pot=None, indent: bool = False) -> str:
    """One rating: label, `now / pot` and a bar (the potential bar is drawn behind the current one)."""
    if now is None:
        return ""
    value = f'<span style="color:{color(now)}">{now}</span>'
    track = ""
    if pot is not None:
        value += f'<i>/</i><span style="color:{color(pot)}">{pot}</span>'
        if pot > now:
            track += f'<div class="pot" style="width:{_width(pot)}%;background:{color(pot)}"></div>'
    track += f'<div class="cur" style="width:{_width(now)}%;background:{color(now)}"></div>'
    return (
        f'<div class="row{" in" if indent else ""}"><span class="lab">{html.escape(label)}</span>'
        f'<span class="val">{value}</span><div class="track">{track}</div></div>'
    )


def _rows(row: pd.Series, spec: list[tuple]) -> str:
    return "".join(rating_row(label, _number(row, now), _number(row, pot), indent) for label, now, pot, indent in spec)


def _pairs(row: pd.Series, spec: list[tuple]) -> str:
    return "".join(rating_row(label, _number(row, column)) for label, column in spec)


def personality(row: pd.Series) -> str:
    high = [label for column, label in TRAITS if row.get(column) == "H"]
    low = [label for column, label in TRAITS if row.get(column) == "L"]
    lines = []
    kind = row.get("Type")
    if isinstance(kind, str) and kind != "Unknown":
        lines.append(f'<div class="text"><span class="lab">Type</span><b>{html.escape(kind)}</b></div>')
    if high:
        lines.append(f'<div class="note"><span class="hi">High:</span> {html.escape(", ".join(high))}</div>')
    if low:
        lines.append(f'<div class="note"><span class="lo">Low:</span> {html.escape(", ".join(low))}</div>')
    prone = row.get("Prone cat")
    if isinstance(prone, str):
        lines.append(f'<div class="text"><span class="lab">Injury proneness</span><b>{html.escape(prone)}</b></div>')
    return "<h5>Personality</h5>" + "".join(lines)


def is_pitcher(slot: str) -> bool:
    return slot in ("SP", "RP")


def full_name(row: pd.Series, fallback: str = "") -> str:
    """First and last name of the player (`fallback`, the short name of the tables, when the file has none)."""
    parts = [str(row[c]).strip() for c in ("First Name", "Last Name") if c in row.index and pd.notna(row[c])]
    return " ".join(p for p in parts if p) or fallback


def headline(row: pd.Series) -> str:
    """Overall and potential rating of the player."""
    parts = [
        f"{label} {_number(row, column)}"
        for label, column in (("Overall", "OVR"), ("Potential", "POT"))
        if _number(row, column)
    ]
    return ", ".join(parts)


def _summary(row: pd.Series) -> str:
    out = ["<h5>Summary</h5>"]
    for label, column in (("Overall", "OVR"), ("Potential", "POT")):
        value = _number(row, column)
        if value is not None:
            out.append(
                f'<div class="big"><span class="lab">{label}</span><b style="color:{color(value)}">{value}</b></div>'
            )
    return "".join(out)


def _hitter_right(row: pd.Series) -> str:
    out = [_summary(row), "<h5>Position ratings</h5>"]
    played = []
    for code, label in POSITIONS:
        now, pot = _number(row, code), _number(row, f"{code} Pot")
        if (now or 0) > 0 or as_20_80(pot or 0) >= POTENTIAL_SHOWN:
            played.append(code)
            out.append(rating_row(label, now or None, pot if pot and pot != now else None))
    groups = []
    if "C" in played:
        groups.append("C")
    if set(played) & {"1B", "2B", "3B", "SS"}:
        groups.append("IF")
    if set(played) & {"LF", "CF", "RF"}:
        groups.append("OF")
    if groups:
        out.append("<h5>Defensive ratings</h5>")
        out += [_pairs(row, DEFENSE[g]) for g in groups]
    return "".join(out)


def _hitter(row: pd.Series) -> tuple[str, str]:
    left = (
        '<h5>Basic batting ratings</h5><div class="sub">Current / Potential</div>'
        + _rows(row, HITTER_BASIC)
        + '<div class="two"><div>'
        + _pairs(row, RUNNING_LEFT)
        + "</div><div>"
        + _pairs(row, RUNNING_RIGHT)
        + "</div></div>"
    )
    style = row.get("BBT")
    if isinstance(style, str):
        left += f'<div class="text"><span class="lab">Batted Ball</span><b>{html.escape(style)}</b></div>'
    return left, _hitter_right(row)


def _kv(label: str, value) -> str:
    if value is None:
        return ""
    return f'<div class="text"><span class="lab">{html.escape(label)}</span><b>{html.escape(str(value))}</b></div>'


def _mph(code) -> str | None:
    return f"{VELOCITY[code]} Mph" if code in VELOCITY else None


def _pitcher(row: pd.Series, slot: str) -> tuple[str, str]:
    left = '<h5>Basic pitching ratings</h5><div class="sub">Current / Potential</div>' + _rows(row, PITCHER_BASIC)
    pitches = [(label, _number(row, c), _number(row, f"{c}P")) for c, label in PITCHES.items()]
    pitches = [(label, now, pot) for label, now, pot in pitches if (now or 0) > 0 or (pot or 0) > 0]
    half = (len(pitches) + 1) // 2
    left += '<div class="two wide" style="margin-top:8px"><div>'
    left += "".join(rating_row(*p) for p in pitches[:half]) + "</div><div>"
    left += "".join(rating_row(*p) for p in pitches[half:]) + "</div></div>"
    left += (
        '<div class="kv"><div>'
        + _kv("GB/FB", GROUND_FLY.get(_number(row, "G/F")))
        + _kv("Arm Slot", ARM_SLOT.get(_number(row, "Slot")))
        + _kv("Velocity", _mph(_number(row, "VELO")))
        + "</div><div>"
        + _kv("Pitcher Type", PITCHER_TYPE.get(_number(row, "PT")))
        + _kv("Suggested Role", "Starter" if slot == "SP" else "Reliever")
        + _kv("Potential", _mph(_number(row, "VT")))
        + "</div></div>"
    )
    right = _summary(row) + "<h5>Position ratings</h5>"
    right += rating_row(
        "Pitcher",
        _number(row, "P") or None,
        _number(row, "P Pot") if _number(row, "P Pot") != _number(row, "P") else None,
    )
    right += "<h5>Other ratings</h5>" + _pairs(
        row, [("Stamina", "STM"), ("Hold Runners", "HLD"), ("Sacrifice Bunt", "BUN")]
    )
    return left, right


def injury_html(row: pd.Series) -> str:
    """One line in the color of the state when the player is injured (day to day, out or on the injured list), else nothing."""
    if str(row.get("INJ", "")).strip().lower() not in ("yes", "y", "1", "true"):
        return ""
    left = str(row.get("Left", "?"))
    state = injuries.status(injuries.days(left), injuries.load_injuries())
    note = " (on the injured list: off the depth charts and the plan)" if state == injuries.IL else ""
    return (
        f'<div class="text" style="color:{injuries.COLORS[state]};font-weight:700">'
        f"Injured: {injuries.LABELS[state]}, {html.escape(left)} left{note}</div>"
    )


def card_html(row: pd.Series, slot: str, title: str = "") -> str:
    """The whole page for a player (a row of player-data) as html with its own style."""
    left, right = _pitcher(row, slot) if is_pitcher(slot) else _hitter(row)
    caption = f'<div class="name">{html.escape(title)}</div>' if title else ""
    caption += injury_html(row)
    return f'{CSS}<div class="pc">{caption}<div>{left}{personality(row)}</div><div>{right}</div></div>'
