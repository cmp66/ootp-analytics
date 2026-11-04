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


# Attribute lists after the stage 3 review (scripts/review_attributes.py, plans/14-models.md). Player-grouped
# cross-validation on Model-Sim 2046-2086: a rating is kept when dropping it (or its skill, for the vs-right / vs-left
# pairs) costs weighted R^2, and added when it gains at least 0.002 and 3 standard errors. League constants, derived
# ratings (MOV, CON), body size, speed (outside baserunning) and age add nothing to the current-value models.
# Every listed rating also exists in the in-game export (TGS27), so the models can score those saves; the batting
# hit-by-pitch rating (+0.002 R^2) is left out because the export does not have it.
REVIEWED_FEATURES = {
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
    ],
    BASERUNNING: ["SPE", "STE", "RUN"],
    PITCHING_FIP: [
        "STU vR",
        "STU vL",
        "CON.1 vR",
        "CON.1 vL",
        "PBABIP vR",
        "PBABIP vL",
        "HRR vR",
        "HRR vL",
        "STM",
        "PIT",
    ],
    PITCHING_BABIP: ["PBABIP vR", "PBABIP vL"],
    FIELDING: {
        2: ["C FRM", "C ABI"],
        3: ["IF RNG", "HT"],
        4: ["IF RNG", "IF ARM", "IF ERR"],
        5: ["IF RNG", "IF ARM", "IF ERR"],
        6: ["IF RNG", "IF ARM", "IF ERR"],
        7: ["OF RNG", "OF ARM"],
        8: ["OF RNG", "OF ARM"],
        9: ["OF RNG", "OF ARM"],
    },
}

# Model family per stream: the simplest family within 0.003 cross-validated weighted R^2 of the best one
# (scripts/compare_models.py, Model-Sim 2046-2086, reviewed attribute lists).
CHOSEN_FAMILY = {
    BATTING: "ridge",
    BASERUNNING: "boosting",
    PITCHING_FIP: "boosting",
    PITCHING_BABIP: "ridge",
    FIELDING: {2: "boosting", 3: "mlp", 4: "mlp", 5: "boosting", 6: "mlp", 7: "boosting", 8: "mlp", 9: "boosting"},
}
