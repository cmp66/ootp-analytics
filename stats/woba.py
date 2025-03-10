# Description: This file contains the functions to calculate
# the wOBA coefficients for a given league.


def calc_league_data(league_totals: dict[str, float]) -> dict[str, float]:

    outs = (
        league_totals["AB"]
        - league_totals["H"]
        + league_totals["CS"]
        + league_totals["GIDP"]
        + league_totals["SF"]
        + league_totals["SH"]
    )
    runs_per_out = league_totals["R"] / outs
    pro_non_outs = league_totals["H"] + league_totals["BB"] + league_totals["HP"]
    unpro_outs = (
        league_totals["AB"]
        - league_totals["H"]
        + league_totals["CS"]
        + league_totals["GIDP"]
    )

    # Run Values
    run_value_bb = 0.14 + runs_per_out

    run_value_hp = 0.025 + run_value_bb
    run_value_1B = 0.155 + run_value_bb
    run_value_2B = 0.3 + run_value_1B
    run_value_3B = 0.27 + run_value_2B
    run_value_hr = 1.4
    run_value_sb = 0.2
    run_value_cs = 0 - ((2 * runs_per_out) + 0.075)

    run_bb = run_value_bb * (league_totals["BB"] - league_totals["IBB"])
    run_hp = run_value_hp * league_totals["HP"]
    run_1B = run_value_1B * league_totals["1B"]
    run_2B = run_value_2B * league_totals["2B"]
    run_3B = run_value_3B * league_totals["3B"]
    run_hr = run_value_hr * league_totals["HR"]
    run_sb = run_value_sb * league_totals["SB"]
    run_cs = run_value_cs * league_totals["CS"]

    total_runs = run_bb + run_hp + run_1B + run_2B + run_3B + run_hr + run_sb + run_cs
    runs_plus = total_runs / pro_non_outs
    runs_minus = total_runs / unpro_outs
    woba_scale = 1 / (runs_plus + runs_minus)

    league_data = {
        "run_value_bb": run_value_bb,
        "run_value_hp": run_value_hp,
        "run_value_1B": run_value_1B,
        "run_value_2B": run_value_2B,
        "run_value_3B": run_value_3B,
        "run_value_hr": run_value_hr,
        "run_value_sb": run_value_sb,
        "run_value_cs": run_value_cs,
        "coef_BB": (run_value_bb + runs_minus) * woba_scale,
        "coef_HP": (run_value_hp + runs_minus) * woba_scale,
        "coef_1B": (run_value_1B + runs_minus) * woba_scale,
        "coef_2B": (run_value_2B + runs_minus) * woba_scale,
        "coef_3B": (run_value_3B + runs_minus) * woba_scale,
        "coef_HR": (run_value_hr + runs_minus) * woba_scale,
        "coef_SB": run_value_sb * woba_scale,
        "coef_CS": run_value_cs * woba_scale,
        "RUNS_PER_OUT": runs_per_out,
        "PRO_NON_OUTS": pro_non_outs,
        "UNPRO_OUTS": unpro_outs,
        "TOTAL_RUNS": total_runs,
        "RUNS_PLUS": runs_plus,
        "RUNS_MINUS": runs_minus,
        "wOBA_SCALE": woba_scale,
    }

    lg_wSB = (
        league_totals["SB"] * run_value_sb + league_totals["CS"] * run_value_cs
    ) / (
        league_totals["BB"]
        + league_totals["HP"]
        + league_totals["1B"]
        - league_totals["IBB"]
    )
    lg_wSB = lg_wSB if lg_wSB > 0 else 0

    lg_woba = (
        league_totals["BB"] * league_data["coef_BB"]
        + league_totals["HP"] * league_data["coef_HP"]
        + league_totals["1B"] * league_data["coef_1B"]
        + league_totals["2B"] * league_data["coef_2B"]
        + league_totals["3B"] * league_data["coef_3B"]
        + league_totals["HR"] * league_data["coef_HR"]
        + lg_wSB
    ) / league_totals["PA"]
    lg_babip = (league_totals["H"] - league_totals["HR"]) / (
        league_totals["AB"]
        - league_totals["HR"]
        - league_totals["SO"]
        + league_totals["SF"]
    )
    lg_xbh_rate = (league_totals["2B"] + league_totals["3B"]) / (
        league_totals["H"] - league_totals["HR"]
    )
    lg_steal_rate = (league_totals["SB"] + league_totals["CS"]) / (
        league_totals["1B"] + league_totals["BB"] + league_totals["HP"]
    )
    lg_sb_rate = league_totals["SB"] / (league_totals["SB"] + league_totals["CS"])

    lg_ubr = (
        league_totals["UBR"]
        / (
            (league_totals["1B"] + league_totals["BB"] + league_totals["HP"]) * 3
            + (league_totals["2B"] * 2)
            + league_totals["3B"]
            - league_totals["CS"] * 3
            - league_totals["SB"]
        )
        if "UBR" in league_totals
        else 0
    )
    lg_3B_rate = league_totals["3B"] / (league_totals["2B"] + league_totals["3B"])
    lg_k_rate = league_totals["SO"] / (
        league_totals["PA"]
        - league_totals["BB"]
        - league_totals["HP"]
        - league_totals["IBB"]
    )
    lg_ubb = (league_totals["BB"] - league_totals["IBB"]) / (
        league_totals["PA"] - league_totals["IBB"] - league_totals["HP"]
    )
    lg_hr_rate = league_totals["HR"] / (
        league_totals["PA"]
        - league_totals["BB"]
        - league_totals["HP"]
        - league_totals["IBB"]
    )
    runs_per_pa = league_totals["R"] / league_totals["PA"]
    lg_hp_rate = league_totals["HP"] / league_totals["PA"]

    lg_avg = league_totals["H"] / league_totals["AB"]
    lg_obp = (league_totals["H"] + league_totals["BB"] + league_totals["HP"]) / (
        league_totals["AB"]
        + league_totals["BB"]
        + league_totals["HP"]
        + league_totals["SF"]
    )
    lg_slug = (
        league_totals["1B"]
        + 2 * league_totals["2B"]
        + 3 * league_totals["3B"]
        + 4 * league_totals["HR"]
    ) / league_totals["AB"]

    league_data.update(
        {
            "lgwSB": lg_wSB,
            "lgwOBA": lg_woba,
            "lgBABIP": lg_babip,
            "lgXBH_RATE": lg_xbh_rate,
            "lgSTEAL_RATE": lg_steal_rate,
            "lgSB_RATE": lg_sb_rate,
            "lgUBR": lg_ubr,
            "lg3B_RATE": lg_3B_rate,
            "lgK_RATE": lg_k_rate,
            "lguBB": lg_ubb,
            "lgHR_RATE": lg_hr_rate,
            "lgRUNS_PER_PA": runs_per_pa,
            "lgHP_RATE": lg_hp_rate,
            "lgAVG": lg_avg,
            "lgOBP": lg_obp,
            "lgSLUG": lg_slug,
        }
    )

    return league_data
