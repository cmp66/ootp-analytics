"""What each model learns: the value streams, their training volumes and the attribute lists.

plans/14-models.md. The BASELINE_FEATURES are the lists the old model classes used, kept so the new trainer can be
compared with the old models on the same footing (stage 2); the reviewed lists replace them in stage 3."""

# Value streams. Each is trained and stored separately.
BATTING = "batting"
BASERUNNING = "baserunning"
PITCHING_FIP = "pitching-fip"
PITCHING_BABIP = "pitching-babip"
FIELDING = "fielding"

RIGHT, LEFT = "right", "left"
SIDES = (RIGHT, LEFT)
FIELD_POSITIONS = (2, 3, 4, 5, 6, 7, 8, 9)

# Minimum volume of a season to be a training row. Reliability grows with volume (year-to-year correlation of
# wRAA600: 0.47 at 50 PA, 0.51 at 100, 0.56 at 200; FIPRuns200 0.41 at 50 IP; fielding 0.48 at 200 IP, 0.59 at 400),
# so low volume rows are also weighted by their volume; the minimum only drops rows that carry almost nothing.
MIN_PA = 100
MIN_IP = 30
MIN_BIP = 100
MIN_FIELDING_IP = 200

# Ratings that exist per side: the name used in the stacked split rows -> (vs right column, vs left column).
BATTING_SIDE_PAIRS = {
    "BA": ("BA vR", "BA vL"),
    "GAP": ("GAP vR", "GAP vL"),
    "POW": ("POW vR", "POW vL"),
    "EYE": ("EYE vR", "EYE vL"),
    "K": ("K vR", "K vL"),
}
PITCHING_SIDE_PAIRS = {
    "STU": ("STU vR", "STU vL"),
    "CON": ("CON.1 vR", "CON.1 vL"),
    "PBABIP": ("PBABIP vR", "PBABIP vL"),
    "HRR": ("HRR vR", "HRR vL"),
}

# Old attribute lists (league constants and all), copied from model/*Model.py for the stage 2 comparison only.
BASELINE_FEATURES = {
    BATTING: [
        "BA vR",
        "BA vL",
        "GAP vR",
        "GAP vL",
        "POW vR",
        "POW vL",
        "EYE vR",
        "EYE vL",
        "K vR",
        "K vL",
        "lgBabipRating",
        "lgAvkRating",
        "lgConRating",
        "lgEyeRating",
        "lgPowRating",
    ],
    BASERUNNING: ["SPE", "SR", "STE", "RUN"],
    PITCHING_FIP: [
        "T",
        "STU vR",
        "STU vL",
        "CON.1 vR",
        "CON.1 vL",
        "PBABIP vR",
        "PBABIP vL",
        "HRR vR",
        "HRR vL",
        "STM",
        "HLD",
        "PIT",
        "lgPbabipRating",
        "lgStuRating",
        "lgHrrRating",
        "lgCtlRating",
        "lgMovRating",
    ],
    FIELDING: {
        9: ["SPE", "OF RNG", "OF ARM", "OF ERR", "lgPbabipRating", "lgRFArm", "lgRFError", "lgRFRange"],
        8: ["SPE", "OF RNG", "OF ARM", "OF ERR", "lgPbabipRating", "lgCFArm", "lgCFError", "lgCFRange"],
        7: ["OF RNG", "OF ARM", "OF ERR", "SPE", "lgPbabipRating", "lgLFArm", "lgLFError", "lgLFRange"],
        6: ["WT", "SPE", "IF RNG", "IF ARM", "TDP", "IF ERR", "lgPbabipRating", "lgSSArm", "lgSSError", "lgSSRange"],
        5: ["WT", "IF RNG", "IF ARM", "TDP", "IF ERR", "lgPbabipRating", "lg3BArm", "lg3BError", "lg3BRange"],
        4: ["WT", "IF RNG", "IF ARM", "TDP", "IF ERR", "lgPbabipRating", "lg2BArm", "lg2BError", "lg2BRange"],
        3: ["IF RNG", "IF ARM", "TDP", "IF ERR", "HT", "lgPbabipRating", "lg1BArm", "lg1BError", "lg1BRange"],
        2: ["C ABI", "C ARM", "C FRM", "lgCBlock", "lgCArm", "lgCFrm"],
    },
}
