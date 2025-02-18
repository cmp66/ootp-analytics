# Description: This file contains the functions to calculate
# the wOBA coefficients for a given league.


def calc_woba_coefficients(league_totals: dict[str, float]) -> dict[str, float]:

    outs = (
        league_totals["AB"]
        - league_totals["H"]
        + league_totals["CS"]
        + league_totals["GIDP"]
        + league_totals["SF"]
        + league_totals["SH"]
    )
    runs_per_out = league_totals["R"] / outs
    pro_non_outs = league_totals["H"] + league_totals["BB"] + league_totals["HBP"]
    unpro_outs = (
        league_totals["AB"]
        - league_totals["H"]
        + league_totals["CS"]
        + league_totals["GIDP"]
    )

    # Run Values
    run_value_bb = 0.14 + runs_per_out

    run_value_hbp = 0.025 + run_value_bb
    run_value_1B = 0.155 + run_value_bb
    run_value_2B = 0.3 + run_value_1B
    run_value_3B = 0.27 + run_value_2B
    run_value_hr = 1.4
    run_value_sb = 0.2
    run_value_cs = 0 - ((2 * runs_per_out) + 0.075)

    run_bb = run_value_bb * (league_totals["BB"] - league_totals["IBB"])
    run_hbp = run_value_hbp * league_totals["HBP"]
    run_1B = run_value_1B * league_totals["1B"]
    run_2B = run_value_2B * league_totals["2B"]
    run_3B = run_value_3B * league_totals["3B"]
    run_hr = run_value_hr * league_totals["HR"]
    run_sb = run_value_sb * league_totals["SB"]
    run_cs = run_value_cs * league_totals["CS"]

    total_runs = run_bb + run_hbp + run_1B + run_2B + run_3B + run_hr + run_sb + run_cs
    runs_plus = total_runs / pro_non_outs
    runs_minus = total_runs / unpro_outs
    woba_scale = 1 / (runs_plus + runs_minus)

    lg_wSB = (
        league_totals["SB"] * run_value_sb + league_totals["CS"] * run_value_cs
    ) / (
        league_totals["H"]
        + league_totals["BB"]
        + league_totals["HBP"]
        - league_totals["IBB"]
    )
    lg_wSB = lg_wSB if lg_wSB > 0 else 0

    # lg_woba = (
    #     league_totals["BB"] * run_bb
    #     + league_totals["HBP"] * run_hbp
    #     + league_totals["1B"] * run_1B
    #     + league_totals["2B"] * run_2B
    #     + league_totals["3B"] * run_3B
    #     + league_totals["HR"] * run_hr
    #     + lg_wSB
    # ) / league_totals["PA"]
    # lg_babip = (league_totals["H"] - league_totals["HR"]) / (
    #     league_totals["AB"]
    #     - league_totals["HR"]
    #     - league_totals["SO"]
    #     + league_totals["SF"]
    # )
    # lg_xbh_rate = (
    #     league_totals["2B"] + league_totals["3B"] + league_totals["HR"]
    # ) / league_totals["H"]
    # lg_steal_rate = (league_totals["SB"] + league_totals["CS"]) / (
    #     league_totals["1B"] + league_totals["BB"] + league_totals["HBP"]
    # )
    # lg_sb_rate = league_totals["SB"] / (league_totals["SB"] + league_totals["CS"])
    # lg_ubr = league_totals["UBR"] / (
    #     (league_totals["1B"] + league_totals["BB"] + league_totals["HBP"]) * 3
    #     + (league_totals["2B"] * 2)
    #     + league_totals["3B"]
    #     - league_totals["CS"] * 3
    #     - league_totals["SB"]
    # )
    # lg_3B_rate = league_totals["3B"] / (league_totals["2B"] + league_totals["3B"])
    # lg_k_rate = league_totals["SO"] / (
    #     league_totals["PA"]
    #     - league_totals["BB"]
    #     - league_totals["HBP"]
    #     - league_totals["IBB"]
    # )
    # lg_ubb = league_totals["BB"] - league_totals["IBB"] / (
    #     league_totals["PA"] - league_totals["IBB"] - league_totals["HBP"]
    # )
    # lg_hr_rate = league_totals["HR"] / (
    #     league_totals["PA"]
    #     - league_totals["BB"]
    #     - league_totals["HBP"]
    #     - league_totals["IBB"]
    # )
    # runs_per_pa = league_totals["R"] / league_totals["PA"]
    # lg_hbp_rate = league_totals["HBP"] / league_totals["PA"]

    woba_coefficients = {
        "BB": (run_value_bb + runs_minus) * woba_scale,
        "HBP": (run_value_hbp + runs_minus) * woba_scale,
        "1B": (run_value_1B + runs_minus) * woba_scale,
        "2B": (run_value_2B + runs_minus) * woba_scale,
        "3B": (run_value_3B + runs_minus) * woba_scale,
        "HR": (run_value_hr + runs_minus) * woba_scale,
        "SB": run_value_sb * woba_scale,
        "CS": run_value_cs * woba_scale,
    }

    return woba_coefficients
