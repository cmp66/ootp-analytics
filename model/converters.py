"""Conversions of the in-game export values (heights, traits, positions...) to numbers; no heavy imports."""

SWITCH = 3  # the value of the batting hand column for a switch hitter

ratings_conversions_80 = {
    -50: -1.000,
    -30: -0.315,
    -25: -0.155,
    -20: -0.75,
    -15: -0.35,
    -10: -0.15,
    -5: -0.05,
    0: 0,
    5: 0.05,
    10: 0.15,
    15: 0.35,
    20: 0.75,
    25: 0.155,
    30: 0.315,
}

ratings_conversions_10 = {
    -5: -155,
    -4: -75,
    -3: -35,
    -2: -15,
    -1: -5,
    0: 0,
    1: 5,
    2: 15,
    3: 35,
    4: 75,
    5: 155,
}


def convert_dump_positions(position: int) -> str:

    if position == 1:
        return "P"
    elif position == 2:
        return "C"
    elif position == 3:
        return "1B"
    elif position == 4:
        return "2B"
    elif position == 5:
        return "3B"
    elif position == 6:
        return "SS"
    elif position == 7:
        return "LF"
    elif position == 8:
        return "CF"
    elif position == 9:
        return "RF"
    elif position == 0:
        return "DH"
    elif position == 10:
        return "DH"
    else:
        raise ValueError(f"Invalid position value: {position}")


def convert_80_rating(rating: int) -> int:
    return float(rating) / 100.0


def convert_10_rating(rating: int) -> int:
    return float(rating) / 10.0


def convert_stars(rating: str) -> int:
    # check for integer first
    try:
        return int(rating)
    except ValueError:
        pass

    if rating == "0.5 Stars":
        return 1
    elif rating == "1.0 Stars":
        return 2
    elif rating == "1.5 Stars":
        return 3
    elif rating == "2.0 Stars":
        return 4
    elif rating == "2.5 Stars":
        return 5
    elif rating == "3.0 Stars":
        return 6
    elif rating == "3.5 Stars":
        return 7
    elif rating == "4.0 Stars":
        return 8
    elif rating == "4.5 Stars":
        return 9
    elif rating == "5.0 Stars":
        return 10
    else:
        return rating


def convert_bbt(bbt: str) -> int:
    if bbt in ["Line Drive"]:
        return 0
    elif bbt in ["Flyball"]:
        return 0
    elif bbt in ["Normal"]:
        return 0
    elif bbt in ["Groundball"]:
        return 0


def convert_gbt(gbt: str) -> int:
    if gbt == "Ex. Pull":
        return 3
    elif gbt == "Pull":
        return 2
    elif gbt == "Normal":
        return 0
    elif gbt == "Spray":
        return 1


def convert_fbt(fbt: str) -> int:
    if fbt in ["Pull"]:
        return 2
    elif fbt in ["Normal"]:
        return 0
    elif fbt in ["Spray"]:
        return 1


def convert_bat_rl(rl: str) -> int:
    if rl == "R":
        return 1
    elif rl == "L":
        return 2
    elif rl == "S":
        return 3


def convert_throws(rl: str) -> int:
    if rl == "R":
        return 1
    elif rl == "L":
        return 2


def convert_groundball_flyball(gbt: str) -> int:
    if gbt == "EX GB":
        return 62
    elif gbt == "GB":
        return 60
    elif gbt == "NEU":
        return 50
    elif gbt == "FB":
        return 46
    elif gbt == "EX FB":
        return 42


def convert_pitch_type(pt: str) -> int:
    if pt == "GB'er":
        return 10
    elif pt == "Normal":
        return 20
    elif pt == "Finesse":
        return 30
    elif pt == "Power":
        return 40


def convert_slot(slot: str) -> int:
    if slot == "OTT":
        return 4
    elif slot == "3/4":
        return 3
    elif slot == "SIDE":
        return 2
    elif slot == "SUB":
        return 1


def convert_velocity(vel: str) -> int:
    if vel == "80-83":
        return 2
    elif vel == "83-85":
        return 3
    elif vel == "84-86":
        return 4
    elif vel == "85-87":
        return 5
    elif vel == "86-88":
        return 6
    elif vel == "87-89":
        return 7
    elif vel == "88-90":
        return 8
    elif vel == "89-91":
        return 9
    elif vel == "90-92":
        return 10
    elif vel == "91-93":
        return 11
    elif vel == "92-94":
        return 12
    elif vel == "93-95":
        return 13
    elif vel == "94-96":
        return 14
    elif vel == "95-97":
        return 15
    elif vel == "96-98":
        return 16
    elif vel == "97-99":
        return 17
    elif vel == "98-100":
        return 18
    elif vel == "99-101":
        return 19
    elif vel == "100+":
        return 20


def convert_personalilty_trait(trait: str) -> int:
    if trait == "1" or trait == "L":
        return 40
    elif trait == "2" or trait == "N" or trait == "U":
        return 100
    elif trait == "3" or trait == "H":
        return 160
    else:
        return trait


def convert_injury_prone(trait: str) -> int:
    if trait == "Durable" or trait == "Iron Man":
        return 40
    elif trait == "Normal":
        return 100
    elif trait == "Fragile" or trait == "Wrecked":
        return 160
    else:
        return trait


def convert_height_to_cm(height: str) -> int:
    height = height.split("'")
    if len(height) != 3:
        raise ValueError("Height must be in the format 'X' Y'")
    return (float(height[0]) * 12.0 + float(height[1])) * 2.54
