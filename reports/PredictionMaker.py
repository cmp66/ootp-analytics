import pandas as pd
from model.HittingModel import HittingModel
from model.PitchingModel import PitchingModel
from model.FieldingModel import FieldingModel
from model.BSRModel import BSRModel
from model.DevModels import (
    DevModel,
    CATEGORY_HITTING,
    CATEGORY_BASERUNNING,
    CATEGORY_SP,
    CATEGORY_RP,
    CATEGORY_C,
    CATEGORY_1B,
    CATEGORY_2B,
    CATEGORY_3B,
    CATEGORY_SS,
    CATEGORY_LF,
    CATEGORY_CF,
    CATEGORY_RF,
)
from model import convert_injury_prone, convert_personalilty_trait
from reports import get_drafted_players_from_statsplus


class PredictionMaker:

    def __init__(self, league, season, model_type):
        self.league = league
        self.season = season
        self.model_type = model_type

    REPORT_COLS = [
        "ID",
        "First Name",
        "Last Name",
        "ORG",
        "Lev",
        "Age",
        "wRAA600",
        "WAA200-SP",
        "WAA200-RP",
        "runsPAdjSeason-P2",
        "runsPAdjSeason-P3",
        "runsPAdjSeason-P4",
        "runsPAdjSeason-P5",
        "runsPAdjSeason-P6",
        "runsPAdjSeason-P7",
        "runsPAdjSeason-P8",
        "runsPAdjSeason-P9",
        "BSR600",
        "WAA600-C",
        "WAA600-1B",
        "WAA600-2B",
        "WAA600-3B",
        "WAA600-SS",
        "WAA600-LF",
        "WAA600-CF",
        "WAA600-RF",
        "MaxWAA600",
        "MaxWAA600Pos",
        "SctCat",
    ]

    fielding_total_wars = [
        "WAA600-C",
        "WAA600-1B",
        "WAA600-2B",
        "WAA600-3B",
        "WAA600-SS",
        "WAA600-LF",
        "WAA600-CF",
        "WAA600-RF",
    ]

    def conform_exported_data(self, df_ratings: pd.DataFrame) -> pd.DataFrame:

        df_ratings.rename(columns={"Prone": "PRONE"}, inplace=True)
        df_ratings["PRONE"] = df_ratings["PRONE"].apply(convert_injury_prone)
        df_ratings["WE"] = df_ratings["WE"].apply(convert_personalilty_trait)
        df_ratings["INT"] = df_ratings["INT"].apply(convert_personalilty_trait)

        return df_ratings

    def calc_offsets(self, use_potential: bool, draftOnly: bool) -> None:

        df_mlb_predictions = self.make_predictions(
            use_potential=use_potential,
            calc_offsets=True,
            primaryposOnly=True,
            draftOnly=draftOnly,
        )

        print("Calculating Offsets")
        print(f"Number of players in predictions: {len(df_mlb_predictions)}")

        # zero_batter_index = df_mlb_predictions["wRAA600"].abs().idxmin()
        # self.zero_batter = df_mlb_predictions.loc[zero_batter_index]["ID"]

        # zero_bsr_index = df_mlb_predictions["BSR600"].abs().idxmin()
        # self.zero_bsr = df_mlb_predictions.loc[zero_bsr_index]["ID"]

        zero_runsP_P2_index = df_mlb_predictions["runsPAdjSeason-P2"].abs().idxmin()
        self.zero_runsP_P2 = df_mlb_predictions.loc[zero_runsP_P2_index]["ID"]

        zero_runsP_P3_index = df_mlb_predictions["runsPAdjSeason-P3"].abs().idxmin()
        self.zero_runsP_P3 = df_mlb_predictions.loc[zero_runsP_P3_index]["ID"]

        zero_runsP_P4_index = df_mlb_predictions["runsPAdjSeason-P4"].abs().idxmin()
        self.zero_runsP_P4 = df_mlb_predictions.loc[zero_runsP_P4_index]["ID"]

        zero_runsP_P5_index = df_mlb_predictions["runsPAdjSeason-P5"].abs().idxmin()
        self.zero_runsP_P5 = df_mlb_predictions.loc[zero_runsP_P5_index]["ID"]

        zero_runsP_P6_index = df_mlb_predictions["runsPAdjSeason-P6"].abs().idxmin()
        self.zero_runsP_P6 = df_mlb_predictions.loc[zero_runsP_P6_index]["ID"]

        zero_runsP_P7_index = df_mlb_predictions["runsPAdjSeason-P7"].abs().idxmin()
        self.zero_runsP_P7 = df_mlb_predictions.loc[zero_runsP_P7_index]["ID"]

        zero_runsP_P8_index = df_mlb_predictions["runsPAdjSeason-P8"].abs().idxmin()
        self.zero_runsP_P8 = df_mlb_predictions.loc[zero_runsP_P8_index]["ID"]

        zero_runsP_P9_index = df_mlb_predictions["runsPAdjSeason-P9"].abs().idxmin()
        self.zero_runsP_P9 = df_mlb_predictions.loc[zero_runsP_P9_index]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA200-SP"] > -400]
        sp_mean = filtered_players["WAA200-SP"].median()
        index_of_mean = filtered_players["WAA200-SP"].sub(sp_mean).abs().idxmin()
        self.zero_sp = df_mlb_predictions.loc[index_of_mean]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA200-RP"] > -400]
        rp_mean = filtered_players["WAA200-RP"].median()
        index_of_mean = filtered_players["WAA200-RP"].sub(rp_mean).abs().idxmin()
        self.zero_rp = df_mlb_predictions.loc[index_of_mean]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA600-C"] > -400]
        c_mean = filtered_players["WAA600-C"].median()
        index_of_mean = filtered_players["WAA600-C"].sub(c_mean).abs().idxmin()
        self.zero_waa600_c = df_mlb_predictions.loc[index_of_mean]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA600-1B"] > -400]
        first_mean = filtered_players["WAA600-1B"].median()
        index_of_mean = filtered_players["WAA600-1B"].sub(first_mean).abs().idxmin()
        self.zero_waa600_1b = df_mlb_predictions.loc[index_of_mean]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA600-2B"] > -400]
        second_mean = filtered_players["WAA600-2B"].median()
        index_of_mean = filtered_players["WAA600-2B"].sub(second_mean).abs().idxmin()
        self.zero_waa600_2b = df_mlb_predictions.loc[index_of_mean]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA600-3B"] > -400]
        third_mean = filtered_players["WAA600-3B"].median()
        index_of_mean = filtered_players["WAA600-3B"].sub(third_mean).abs().idxmin()
        self.zero_waa600_3b = df_mlb_predictions.loc[index_of_mean]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA600-SS"] > -400]
        ss_mean = filtered_players["WAA600-SS"].median()
        index_of_mean = filtered_players["WAA600-SS"].sub(ss_mean).abs().idxmin()
        self.zero_waa600_ss = df_mlb_predictions.loc[index_of_mean]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA600-LF"] > -400]
        lf_mean = filtered_players["WAA600-LF"].median()
        index_of_mean = filtered_players["WAA600-LF"].sub(lf_mean).abs().idxmin()
        self.zero_waa600_lf = df_mlb_predictions.loc[index_of_mean]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA600-CF"] > -400]
        cf_mean = filtered_players["WAA600-CF"].median()
        index_of_mean = filtered_players["WAA600-CF"].sub(cf_mean).abs().idxmin()
        self.zero_waa600_cf = df_mlb_predictions.loc[index_of_mean]["ID"]

        filtered_players = df_mlb_predictions[df_mlb_predictions["WAA600-RF"] > -400]
        rf_mean = filtered_players["WAA600-RF"].median()
        index_of_mean = filtered_players["WAA600-RF"].sub(rf_mean).abs().idxmin()
        self.zero_waa600_rf = df_mlb_predictions.loc[index_of_mean]["ID"]

        print(f"zero_waa600_sp: {self.zero_sp} {sp_mean}")
        print(f"zero_waa600_rp: {self.zero_rp} {rp_mean}")
        print(f"zero_waa600_c: {self.zero_waa600_c} {c_mean}")
        print(f"zero_waa600_1b: {self.zero_waa600_1b} {first_mean}")
        print(f"zero_waa600_2b: {self.zero_waa600_2b} {second_mean}")
        print(f"zero_waa600_3b: {self.zero_waa600_3b} {third_mean}")
        print(f"zero_waa600_ss: {self.zero_waa600_ss} {ss_mean}")
        print(f"zero_waa600_lf: {self.zero_waa600_lf} {lf_mean}")
        print(f"zero_waa600_cf: {self.zero_waa600_cf}  {cf_mean}")
        print(f"zero_waa600_rf: {self.zero_waa600_rf} {rf_mean}")

    def apply_attribute_offsets(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        # hitting_offset = df_predictions.loc[df_predictions["ID"] == self.zero_batter]["wRAA600"].values[0]
        # df_predictions.loc[:, "wRAA600"] = df_predictions["wRAA600"].apply(lambda x: x - hitting_offset if x != -500 else x)

        # bsr_offset = df_predictions.loc[df_predictions["ID"] == self.zero_bsr]["BSR600"].values[0]
        # df_predictions.loc[:, "BSR600"] = df_predictions["BSR600"].apply(lambda x: x - bsr_offset if x != -500 else x)

        runsP_P2_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_runsP_P2
        ]["runsPAdjSeason-P2"].values[0]
        df_predictions.loc[:, "runsPAdjSeason-P2"] = df_predictions[
            "runsPAdjSeason-P2"
        ].apply(lambda x: x - runsP_P2_offset if x != -500 else x)

        runsP_P3_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_runsP_P3
        ]["runsPAdjSeason-P3"].values[0]
        df_predictions.loc[:, "runsPAdjSeason-P3"] = df_predictions[
            "runsPAdjSeason-P3"
        ].apply(lambda x: x - runsP_P3_offset if x != -500 else x)

        runsP_P4_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_runsP_P4
        ]["runsPAdjSeason-P4"].values[0]
        df_predictions.loc[:, "runsPAdjSeason-P4"] = df_predictions[
            "runsPAdjSeason-P4"
        ].apply(lambda x: x - runsP_P4_offset if x != -500 else x)

        runsP_P5_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_runsP_P5
        ]["runsPAdjSeason-P5"].values[0]
        df_predictions.loc[:, "runsPAdjSeason-P5"] = df_predictions[
            "runsPAdjSeason-P5"
        ].apply(lambda x: x - runsP_P5_offset if x != -500 else x)

        runsP_P6_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_runsP_P6
        ]["runsPAdjSeason-P6"].values[0]
        df_predictions.loc[:, "runsPAdjSeason-P6"] = df_predictions[
            "runsPAdjSeason-P6"
        ].apply(lambda x: x - runsP_P6_offset if x != -500 else x)

        runsP_P7_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_runsP_P7
        ]["runsPAdjSeason-P7"].values[0]
        df_predictions.loc[:, "runsPAdjSeason-P7"] = df_predictions[
            "runsPAdjSeason-P7"
        ].apply(lambda x: x - runsP_P7_offset if x != -500 else x)

        runsP_P8_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_runsP_P8
        ]["runsPAdjSeason-P8"].values[0]
        df_predictions.loc[:, "runsPAdjSeason-P8"] = df_predictions[
            "runsPAdjSeason-P8"
        ].apply(lambda x: x - runsP_P8_offset if x != -500 else x)

        runsP_P9_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_runsP_P9
        ]["runsPAdjSeason-P9"].values[0]
        df_predictions.loc[:, "runsPAdjSeason-P9"] = df_predictions[
            "runsPAdjSeason-P9"
        ].apply(lambda x: x - runsP_P9_offset if x != -500 else x)

        return df_predictions

    def apply_positional_offsets(self, df_predictions: pd.DataFrame) -> pd.DataFrame:

        sp_offset = df_predictions.loc[df_predictions["ID"] == self.zero_sp][
            "WAA200-SP"
        ].values[0]
        df_predictions.loc[:, "WAA200-SP"] = df_predictions["WAA200-SP"].apply(
            lambda x: x - sp_offset if x != -500 else x
        )

        rp_offset = df_predictions.loc[df_predictions["ID"] == self.zero_rp][
            "WAA200-RP"
        ].values[0]
        df_predictions.loc[:, "WAA200-RP"] = df_predictions["WAA200-RP"].apply(
            lambda x: x - rp_offset if x != -500 else x
        )

        waa600_c_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_waa600_c
        ]["WAA600-C"].values[0]
        df_predictions.loc[:, "WAA600-C"] = df_predictions["WAA600-C"].apply(
            lambda x: x - waa600_c_offset if x != -500 else x
        )

        waa600_1b_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_waa600_1b
        ]["WAA600-1B"].values[0]
        df_predictions.loc[:, "WAA600-1B"] = df_predictions["WAA600-1B"].apply(
            lambda x: x - waa600_1b_offset if x != -500 else x
        )

        waa600_2b_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_waa600_2b
        ]["WAA600-2B"].values[0]
        df_predictions.loc[:, "WAA600-2B"] = df_predictions["WAA600-2B"].apply(
            lambda x: x - waa600_2b_offset if x != -500 else x
        )

        waa600_3b_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_waa600_3b
        ]["WAA600-3B"].values[0]
        df_predictions.loc[:, "WAA600-3B"] = df_predictions["WAA600-3B"].apply(
            lambda x: x - waa600_3b_offset if x != -500 else x
        )

        waa600_ss_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_waa600_ss
        ]["WAA600-SS"].values[0]
        df_predictions.loc[:, "WAA600-SS"] = df_predictions["WAA600-SS"].apply(
            lambda x: x - waa600_ss_offset if x != -500 else x
        )

        waa600_lf_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_waa600_lf
        ]["WAA600-LF"].values[0]
        df_predictions.loc[:, "WAA600-LF"] = df_predictions["WAA600-LF"].apply(
            lambda x: x - waa600_lf_offset if x != -500 else x
        )

        waa600_cf_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_waa600_cf
        ]["WAA600-CF"].values[0]
        df_predictions.loc[:, "WAA600-CF"] = df_predictions["WAA600-CF"].apply(
            lambda x: x - waa600_cf_offset if x != -500 else x
        )

        waa600_rf_offset = df_predictions.loc[
            df_predictions["ID"] == self.zero_waa600_rf
        ]["WAA600-RF"].values[0]
        df_predictions.loc[:, "WAA600-RF"] = df_predictions["WAA600-RF"].apply(
            lambda x: x - waa600_rf_offset if x != -500 else x
        )

        return df_predictions

    def make_predictions(
        self,
        use_potential: bool,
        calc_offsets=False,
        draftOnly=False,
        primaryposOnly=False,
        org_filter=None,
        iafa=False,
    ) -> pd.DataFrame:
        use_primary = primaryposOnly
        df_ratings = pd.read_csv(
            f"./files/{self.league}/{self.season}/output/{self.league}-{self.season}-player-data.csv",
            low_memory=False,
        )
        woba = pd.read_csv(
            f"./files/{self.league}/{self.season}/output/{self.league}-{self.season}-woba-calcs.csv"
        )

        woba.set_index("Stat", inplace=True)

        if draftOnly:
            df_baseline_players = df_ratings[
                (df_ratings["Lev"] == "R+")
                | (df_ratings["Lev"] == "R")
                | (df_ratings["Lev"] == "R-")
                | (df_ratings["Lev"] == "WL")
                | (df_ratings["Lev"] == "A-")
            ]
        else:
            df_baseline_players = df_ratings[
                (df_ratings["Lev"] == "MLB")
                | ((df_ratings["Lev"] == "FA") & (df_ratings["OVR"] >= 30))
            ]

        if calc_offsets:
            if use_potential:
                df_baseline_players = df_baseline_players[
                    (df_baseline_players["Age"] <= 25)
                ]
            use_primary = True

        if draftOnly:
            df_non_baseline_players = df_ratings[
                (df_ratings["Lev"] == "R+")
                | (df_ratings["Lev"] == "R")
                | (df_ratings["Lev"] == "R-")
                | (df_ratings["Lev"] == "WL")
                | (df_ratings["Lev"] == "A-")
                | (df_ratings["SctCat"] == "Amateur")
            ]
        else:
            df_non_baseline_players = df_ratings[
                (df_ratings["Lev"] == "MLB")
                | (df_ratings["Lev"] == "FA")
                | (df_ratings["Lev"] == "AAA")
                | (df_ratings["Lev"] == "AA")
                | (df_ratings["Lev"] == "A+")
                | (df_ratings["Lev"] == "A")
                | (df_ratings["Lev"] == "A-")
                | (df_ratings["Lev"] == "R+")
                | (df_ratings["Lev"] == "R")
                | (df_ratings["Lev"] == "S A")
                | (df_ratings["Lev"] == "INT")
                | (df_ratings["Lev"] == "R-")
                | (df_ratings["Lev"] == "WL")
            ]

        if self.model_type == "Standard":
            df_non_baseline_players = df_non_baseline_players[
                df_non_baseline_players["POT"] >= 30
            ]

        if use_potential:
            df_baseline_players = df_baseline_players[
                (df_baseline_players["Age"] <= 25)
            ]
            df_non_baseline_players = df_non_baseline_players[
                (df_non_baseline_players["Age"] <= 25)
            ]

        df_baseline_players = self.conform_exported_data(
            pd.DataFrame(df_baseline_players)
        )
        df_non_baseline_players = self.conform_exported_data(
            pd.DataFrame(df_non_baseline_players)
        )

        df_baseline_players = df_baseline_players[df_baseline_players["ID"].notnull()]
        df_non_baseline_players = df_non_baseline_players[
            df_non_baseline_players["ID"].notnull()
        ]

        df_ratings_processed = df_baseline_players.copy(deep=True)

        print(f"Number of players in baseline: {len(df_baseline_players)}")
        print(f"Number of players in non-baseline: {len(df_non_baseline_players)}")

        print("Processing Baseline Players")
        df_ratings_processed = pd.merge(
            df_ratings_processed,
            self.predict_batting(
                df_baseline_players.copy(deep=True), woba, use_potential
            ),
            how="outer",
            on="ID",
        )
        df_ratings_processed = pd.merge(
            df_ratings_processed,
            self.predict_pitching(
                df_baseline_players.copy(deep=True), woba, use_potential
            ),
            how="outer",
            on="ID",
        )
        df_ratings_processed = pd.merge(
            df_ratings_processed,
            self.predict_fielding(
                df_baseline_players.copy(deep=True),
                use_primary,
                org_filter=org_filter,
                use_potential=use_potential,
            ),
            how="outer",
            on="ID",
        )
        df_ratings_processed = pd.merge(
            df_ratings_processed,
            self.predict_baserunning(
                df_baseline_players.copy(deep=True), use_potential
            ),
            how="outer",
            on="ID",
        )

        if not calc_offsets:
            print("Processing Non-Baseline Players")
            df_ratings_processed = df_non_baseline_players.copy(deep=True)
            df_ratings_processed = pd.merge(
                df_ratings_processed,
                self.predict_batting(
                    df_non_baseline_players.copy(deep=True), woba, use_potential
                ),
                how="outer",
                on="ID",
            )
            df_ratings_processed = pd.merge(
                df_ratings_processed,
                self.predict_pitching(
                    df_non_baseline_players.copy(deep=True), woba, use_potential
                ),
                how="outer",
                on="ID",
            )
            df_ratings_processed = pd.merge(
                df_ratings_processed,
                self.predict_fielding(
                    df_non_baseline_players.copy(deep=True),
                    use_primary,
                    org_filter=org_filter,
                    use_potential=use_potential,
                ),
                how="outer",
                on="ID",
            )
            df_ratings_processed = pd.merge(
                df_ratings_processed,
                self.predict_baserunning(
                    df_non_baseline_players.copy(deep=True), use_potential
                ),
                how="outer",
                on="ID",
            )

        print("Calculating Positional Ratings")

        df_ratings_processed = self.calc_all_potential_positional_ratings(
            df_ratings_processed
        )

        if not calc_offsets:
            df_ratings_processed = self.apply_positional_offsets(df_ratings_processed)

        df_ratings_processed = self.calc_best_position(df_ratings_processed)

        fix_values = {"MaxWAA600Pos": "NA"}
        df_ratings_processed.fillna(fix_values, inplace=True)
        df_ratings_processed.fillna(-500, inplace=True)
        final_predictions = df_ratings_processed[self.REPORT_COLS]

        final_predictions = final_predictions[
            (final_predictions["MaxWAA600"] != -500)
            | (final_predictions["WAA200-SP"] != -500)
            | (final_predictions["WAA200-RP"] != -500)
        ]

        if draftOnly and not calc_offsets:
            final_predictions = final_predictions[
                final_predictions["SctCat"] == "Amateur"
            ]
            if org_filter == "FA":
                drafted_players = get_drafted_players_from_statsplus(self.league)
                final_predictions = final_predictions[
                    ~final_predictions["ID"].isin(drafted_players)
                ]
            else:
                drafted_players = pd.read_csv(
                    f"./files/{self.league}/{self.season}/draft/draft.csv"
                )["ID"].tolist()
                final_predictions = final_predictions[
                    final_predictions["ID"].isin(drafted_players)
                ]

        elif iafa:
            final_predictions = final_predictions[
                final_predictions["SctCat"] == "International"
            ]
        else:
            final_predictions = final_predictions[
                final_predictions["SctCat"] != "Amateur"
            ]

        final_predictions.drop(columns=["SctCat"], inplace=True)

        return final_predictions

    def predict_batting(
        self, df_ratings: pd.DataFrame, woba: pd.DataFrame, use_potential: bool
    ) -> pd.DataFrame:

        df_ratings_temp = df_ratings.copy(deep=True)

        df_ratings_process = df_ratings_temp

        lgwOBA = woba.loc["lgwOBA"]["Value"]
        df_ratings_process["lgwOBA"] = lgwOBA

        lgOBP = woba.loc["lgOBP"]["Value"]
        df_ratings_process["lgOBP"] = lgOBP

        # jam a dummy value in for now
        df_ratings_process["wRAA600"] = 0

        if use_potential:
            hittingModel = DevModel(
                league=self.league,
                season_start=self.season,
                season_end=self,
                ratings_type=self.model_type,
                category=CATEGORY_HITTING,
            )
        else:
            hittingModel = HittingModel(
                league=self.league,
                season_start=self.season,
                season_end=self.season,
                ratings_type=self.model_type,
                vsType="potential" if use_potential else "total",
            )

        # if use_potential:
        #    for k,v in hitting_conversion_to_potential.items():
        #            df_ratings_process[k] = df_ratings_process[v]

        # df_ratings_process = df_ratings_process[feature_values["total"] + [targets["total"][0]] + ['ID']]

        # print(df_ratings_process.loc[df_ratings_process.isnull().any(axis=1)])

        hittingModel.load_model()

        predictions = hittingModel.predict(
            self.season, skip_load=True, preloaded_data=df_ratings_process
        )

        predictions.rename(columns={"Predictions": "wRAA600"}, inplace=True)
        return predictions[["ID", "wRAA600"]]

    def predict_pitching(
        self, df_ratings: pd.DataFrame, woba: pd.DataFrame, use_potential: bool
    ) -> pd.DataFrame:

        df_ratings_temp = df_ratings.copy(deep=True)

        df_ratings_processed = (
            df_ratings_temp  # .merge(df_ratings_temp, on="ID", how="left")
        )
        for role in ["SP", "RP"]:

            df_ratings_process = df_ratings_temp.copy(deep=True)

            if role == "SP":
                df_ratings_process = df_ratings_process[
                    df_ratings_process["LPOS"] == "SP"
                ]
            else:
                df_ratings_process = df_ratings_process[
                    (df_ratings_process["LPOS"] == "RP")
                    | (df_ratings_process["LPOS"] == "CL")
                ]

            # check if dataframe is empty
            if df_ratings_process.empty:
                df_ratings_processed.loc[:, "WAA200-" + role] = -500
                continue

            runs_per_out = woba.loc["RUNS_PER_OUT"]["Value"]
            df_ratings_process["RUNS_PER_OUT"] = runs_per_out

            lgwOBA = woba.loc["lgwOBA"]["Value"]
            df_ratings_process["lgwOBA"] = lgwOBA

            # jam a dummy value in for now
            df_ratings_process["WAA200"] = 0

            if use_potential:
                pitchingModel = DevModel(
                    league=self.league,
                    season_start=self.season,
                    season_end=self,
                    ratings_type=self.model_type,
                    category=CATEGORY_SP if role == "SP" else CATEGORY_RP,
                )
            else:
                pitchingModel = PitchingModel(
                    league=self.league,
                    season_start=self.season,
                    season_end=self.season,
                    role=role,
                    ratings_type=self.model_type,
                    use_potential=use_potential,
                )

            pitchingModel.load_model()
            predictions = pitchingModel.predict(
                self.season, skip_load=True, preloaded_data=df_ratings_process
            )
            predictions = predictions.sort_values(by="Predictions", ascending=False)

            df_ratings_processed.loc[:, "WAA200-" + role] = predictions["Predictions"]

        return df_ratings_processed[["ID", "WAA200-SP", "WAA200-RP"]]

    def calc_fielding_mins_primary(
        self, df_ratings: pd.DataFrame, position: int
    ) -> pd.DataFrame:

        if position == 2:
            df_filtered = df_ratings[df_ratings["LPOS"] == "C"]
        elif position == 3:
            df_filtered = df_ratings[df_ratings["LPOS"] == "1B"]
        elif position == 4:
            df_filtered = df_ratings[df_ratings["LPOS"] == "2B"]
        elif position == 5:
            df_filtered = df_ratings[df_ratings["LPOS"] == "3B"]
        elif position == 6:
            df_filtered = df_ratings[df_ratings["LPOS"] == "SS"]
        elif position == 7:
            df_filtered = df_ratings[df_ratings["LPOS"] == "LF"]
        elif position == 8:
            df_filtered = df_ratings[df_ratings["LPOS"] == "CF"]
        elif position == 9:
            df_filtered = df_ratings[df_ratings["LPOS"] == "RF"]

        return df_filtered

    def calc_fielding_mins_all(
        self, df_ratings: pd.DataFrame, position: int, org_filter: str
    ) -> pd.DataFrame:

        # filter = df_fielding_playables.iloc[:, 0] == position
        if position == 2:
            df_filtered = df_ratings[
                ((df_ratings["C"] > 0) & (df_ratings["ORG"] == org_filter))
                | (df_ratings["LPOS"] == "C")
            ]
        elif position == 3:
            df_filtered = df_ratings[
                ((df_ratings["1B"] > 0) & (df_ratings["ORG"] == org_filter))
                | (df_ratings["LPOS"] == "1B")
            ]
        elif position == 4:
            df_filtered = df_ratings[
                ((df_ratings["2B"] > 0) & (df_ratings["ORG"] == org_filter))
                | (df_ratings["LPOS"] == "2B")
            ]
        elif position == 5:
            df_filtered = df_ratings[
                ((df_ratings["3B"] > 0) & (df_ratings["ORG"] == org_filter))
                | (df_ratings["LPOS"] == "3B")
            ]
        elif position == 6:
            df_filtered = df_ratings[
                ((df_ratings["SS"] > 0) & (df_ratings["ORG"] == org_filter))
                | (df_ratings["LPOS"] == "SS")
            ]
        elif position == 7:
            df_filtered = df_ratings[
                ((df_ratings["LF"] > 0) & (df_ratings["ORG"] == org_filter))
                | (df_ratings["LPOS"] == "LF")
            ]
        elif position == 8:
            df_filtered = df_ratings[
                ((df_ratings["CF"] > 0) & (df_ratings["ORG"] == org_filter))
                | (df_ratings["LPOS"] == "CF")
            ]
        elif position == 9:
            df_filtered = df_ratings[
                ((df_ratings["RF"] > 0) & (df_ratings["ORG"] == org_filter))
                | (df_ratings["LPOS"] == "RF")
            ]
        return df_filtered

    def predict_fielding(
        self,
        df_ratings: pd.DataFrame,
        primaryPosOnly=False,
        org_filter=None,
        use_potential=False,
    ) -> pd.DataFrame:

        category_conversion = {
            2: CATEGORY_C,
            3: CATEGORY_1B,
            4: CATEGORY_2B,
            5: CATEGORY_3B,
            6: CATEGORY_SS,
            7: CATEGORY_LF,
            8: CATEGORY_CF,
            9: CATEGORY_RF,
        }

        df_ratings_temp = df_ratings.copy(deep=True)

        df_ratings_process = df_ratings_temp.copy(deep=True)
        df_ratings_processed = df_ratings_process.copy(deep=True)

        for position in [2, 3, 4, 5, 6, 7, 8, 9]:

            df_ratings_process = df_ratings_temp.copy(deep=True)

            if primaryPosOnly:
                df_ratings_process = self.calc_fielding_mins_primary(
                    df_ratings_process, position
                )
            else:
                df_ratings_process = self.calc_fielding_mins_all(
                    df_ratings_process, position, org_filter=org_filter
                )

            if df_ratings_process.empty:
                df_ratings_processed["runsPAdjSeason-P" + str(position)] = -500
                continue

            # jam a dummy value in for now
            df_ratings_process["runsPAdjSeason"] = 0

            if use_potential:
                fieldingModel = DevModel(
                    league=self.league,
                    season_start=self.season,
                    season_end=self,
                    ratings_type=self.model_type,
                    category=category_conversion[position],
                )
            else:
                fieldingModel = FieldingModel(
                    league=self.league,
                    season_start=self.season,
                    season_end=self.season,
                    position=position,
                    ratings_type=self.model_type,
                )

            fieldingModel.load_model()
            predictions = fieldingModel.predict(
                self.season, skip_load=True, preloaded_data=df_ratings_process
            )
            predictions = predictions.sort_values(by="Predictions", ascending=False)

            df_ratings_processed["runsPAdjSeason-P" + str(position)] = predictions[
                "Predictions"
            ]

        return df_ratings_processed[
            [
                "ID",
                "runsPAdjSeason-P2",
                "runsPAdjSeason-P3",
                "runsPAdjSeason-P4",
                "runsPAdjSeason-P5",
                "runsPAdjSeason-P6",
                "runsPAdjSeason-P7",
                "runsPAdjSeason-P8",
                "runsPAdjSeason-P9",
            ]
        ]

    def predict_baserunning(
        self, df_ratings: pd.DataFrame, use_potential=False
    ) -> pd.DataFrame:

        df_ratings_temp = df_ratings.copy(deep=True)

        df_ratings_process = df_ratings_temp

        # jam a dummy value in for now
        df_ratings_process["BSR600"] = 0

        if use_potential:
            bsrModel = DevModel(
                league=self.league,
                season_start=self.season,
                season_end=self,
                ratings_type=self.model_type,
                category=CATEGORY_BASERUNNING,
            )
        else:
            bsrModel = BSRModel(
                league=self.league,
                season_start=self.season,
                season_end=self.season,
                ratings_type=self.model_type,
                scale="10",
            )

        bsrModel.load_model()
        predictions = bsrModel.predict(
            self.season, skip_load=True, preloaded_data=df_ratings_process
        )

        df_ratings_process["BSR600"] = predictions["Predictions"]

        return df_ratings_process[["ID", "BSR600"]]

    def calc_all_potential_positional_ratings(
        self, df_ratings: pd.DataFrame
    ) -> pd.DataFrame:

        df_ratings["WAA600-C"] = (
            df_ratings["wRAA600"]
            + df_ratings["BSR600"]
            + df_ratings["runsPAdjSeason-P2"]
        ) / 10.0
        df_ratings["WAA600-1B"] = (
            df_ratings["wRAA600"]
            + df_ratings["BSR600"]
            + df_ratings["runsPAdjSeason-P3"]
        ) / 10.0
        df_ratings["WAA600-2B"] = (
            df_ratings["wRAA600"]
            + df_ratings["BSR600"]
            + df_ratings["runsPAdjSeason-P4"]
        ) / 10.0
        df_ratings["WAA600-3B"] = (
            df_ratings["wRAA600"]
            + df_ratings["BSR600"]
            + df_ratings["runsPAdjSeason-P5"]
        ) / 10.0
        df_ratings["WAA600-SS"] = (
            df_ratings["wRAA600"]
            + df_ratings["BSR600"]
            + df_ratings["runsPAdjSeason-P6"]
        ) / 10.0
        df_ratings["WAA600-LF"] = (
            df_ratings["wRAA600"]
            + df_ratings["BSR600"]
            + df_ratings["runsPAdjSeason-P7"]
        ) / 10.0
        df_ratings["WAA600-CF"] = (
            df_ratings["wRAA600"]
            + df_ratings["BSR600"]
            + df_ratings["runsPAdjSeason-P8"]
        ) / 10.0
        df_ratings["WAA600-RF"] = (
            df_ratings["wRAA600"]
            + df_ratings["BSR600"]
            + df_ratings["runsPAdjSeason-P9"]
        ) / 10.0

        df_ratings[self.fielding_total_wars] = df_ratings[
            self.fielding_total_wars
        ].fillna(-500)

        return df_ratings

    def calc_best_position(self, df_ratings: pd.DataFrame) -> str:

        df_ratings["MaxWAA600"] = df_ratings[self.fielding_total_wars].max(axis=1)
        df_ratings["MaxWAA600Column"] = df_ratings[self.fielding_total_wars].idxmax(
            axis=1
        )
        df_ratings["MaxWAA600Pos"] = df_ratings["MaxWAA600Column"].str.extract(
            r"WAA600-(.*)"
        )[0]

        return df_ratings
