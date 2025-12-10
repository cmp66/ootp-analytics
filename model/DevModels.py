import pandas as pd
from model import Modeler
from model.HittingModel import HittingModel
from model.BSRModel import BSRModel
from model.PitchingModel import PitchingModel
from model.FieldingModel import FieldingModel

DEV_AGE_TARGET = 25
DEV_PREDICTION_AGE_MAX = 23

CATEGORY_HITTING = "hitting"
CATEGORY_BASERUNNING = "baserunning"
CATEGORY_SP = "starter"
CATEGORY_RP = "reliever"
CATEGORY_C = "catcher"
CATEGORY_1B = "firstbase"
CATEGORY_2B = "secondbase"
CATEGORY_3B = "thirdbase"
CATEGORY_SS = "shortstop"
CATEGORY_LF = "leftfield"
CATEGORY_CF = "centerfield"
CATEGORY_RF = "rightfield"

feature_values = {
    CATEGORY_HITTING: [
        "BABIP",
        "BA vR",
        "BA vL",
        "BA P",
        "GAP",
        "GAP vR",
        "GAP vL",
        "GAP P",
        "POW",
        "POW vR",
        "POW vL",
        "POW P",
        "EYE",
        "EYE vR",
        "EYE vL",
        "EYE P",
        "K's",
        "K vR",
        "K vL",
        "K P",
        "Age",
        # "LEA",
        # "LOY",
        # "FIN",
        "WE",
        "INT",
        # "PRONE",
        # "WT",
        "SPE",
        # "RUN",
    ],
    CATEGORY_BASERUNNING: [
        "Age",
        "SPE",
        "SR",
        "STE",
        "RUN",
        "WT",
        "WE",
        # "INT",
        # "PRONE"
    ],
    CATEGORY_SP: [
        "Age",
        "T",
        "STU",
        "STU vR",
        "STU vL",
        "CON.1",
        "CON.1 vR",
        "CON.1 vL",
        "PBABIP",
        "PBABIP vR",
        "PBABIP vL",
        "HRR",
        "HRR vR",
        "HRR vL",
        "VELO",
        "STM",
        "HLD",
        "PIT",
        # "HT",
        # "Slot",
        "STU P",
        "CON.1 P",
        "PBABIP P",
        "HRR P",
        "WE",
        "INT",
        "PRONE",
    ],
    CATEGORY_RP: [
        "Age",
        "T",
        "STU",
        "STU vR",
        "STU vL",
        "CON.1",
        "CON.1 vR",
        "CON.1 vL",
        "PBABIP",
        "PBABIP vR",
        "PBABIP vL",
        "HRR",
        "HRR vR",
        "HRR vL",
        "PIT",
        "STM",
        "HLD",
        "VELO",
        # "HT",
        # "Slot",
        "STU P",
        "CON.1 P",
        "PBABIP P",
        "HRR P",
        "WE",
        "INT",
        # "PRONE",
    ],
    CATEGORY_RF: [
        "Age",
        "WE",
        "INT",
        "PRONE",
        "WT",
        "SPE",
        "OF RNG",
        "OF ARM",
        "OF ERR",
        "RF",
    ],
    CATEGORY_CF: [
        "Age",
        "WE",
        "INT",
        "PRONE",
        "WT",
        "SPE",
        "OF RNG",
        "OF ARM",
        "OF ERR",
        "CF",
    ],
    CATEGORY_LF: [
        "Age",
        "WE",
        "INT",
        "PRONE",
        "WT",
        "SPE",
        "OF RNG",
        "OF ARM",
        "OF ERR",
        "LF",
    ],
    CATEGORY_SS: [
        "Age",
        "WE",
        "INT",
        "PRONE",
        "WT",
        "SPE",
        "IF RNG",
        "IF ARM",
        "TDP",
        "IF ERR",
        "SS",
    ],
    CATEGORY_3B: [
        "Age",
        "WE",
        "INT",
        "PRONE",
        "WT",
        "SPE",
        "IF RNG",
        "IF ARM",
        "TDP",
        "IF ERR",
        "3B",
    ],
    CATEGORY_2B: [
        "Age",
        "WE",
        "INT",
        "PRONE",
        "WT",
        "SPE",
        "IF RNG",
        "IF ARM",
        "TDP",
        "IF ERR",
        "2B",
    ],
    CATEGORY_1B: [
        "Age",
        "WE",
        "INT",
        "PRONE",
        "WT",
        "SPE",
        "IF RNG",
        "IF ARM",
        "TDP",
        "IF ERR",
        "HT",
        "1B",
    ],
    CATEGORY_C: ["Age", "WE", "INT", "PRONE", "WT", "C", "C ABI", "C ARM", "C FRM"],
}

exclude_adj = [
    "ID",
    "lgwOBA",
    "lgOBP",
    "lgBABIP",
    "lgHR_RATE",
    "lgK_RATE",
    "lgXBH_RATE",
    "wOBA_SCALE",
    "B",
    "SPE",
]

targets = {
    CATEGORY_HITTING: ["wRAA600-DEV"],
    CATEGORY_BASERUNNING: ["BSR600-DEV"],
    CATEGORY_SP: ["WAA200"],
    CATEGORY_RP: ["WAA200"],
    CATEGORY_RF: ["runsPAdjSeason-DEV"],
    CATEGORY_CF: ["runsPAdjSeason-DEV"],
    CATEGORY_LF: ["runsPAdjSeason-DEV"],
    CATEGORY_SS: ["runsPAdjSeason-DEV"],
    CATEGORY_3B: ["runsPAdjSeason-DEV"],
    CATEGORY_2B: ["runsPAdjSeason-DEV"],
    CATEGORY_1B: ["runsPAdjSeason-DEV"],
    CATEGORY_C: ["runsPAdjSeason-DEV"],
}

prediction_targets = {
    CATEGORY_HITTING: ["wRAA600"],
    CATEGORY_BASERUNNING: ["BSR600"],
    CATEGORY_SP: ["WAA200"],
    CATEGORY_RP: ["WAA200"],
    CATEGORY_RF: ["runsPAdjSeason"],
    CATEGORY_CF: ["runsPAdjSeason"],
    CATEGORY_LF: ["runsPAdjSeason"],
    CATEGORY_SS: ["runsPAdjSeason"],
    CATEGORY_3B: ["runsPAdjSeason"],
    CATEGORY_2B: ["runsPAdjSeason"],
    CATEGORY_1B: ["runsPAdjSeason"],
    CATEGORY_C: ["runsPAdjSeason"],
}

fielding_conversions = {
    CATEGORY_RF: 9,
    CATEGORY_CF: 8,
    CATEGORY_LF: 7,
    CATEGORY_SS: 6,
    CATEGORY_3B: 5,
    CATEGORY_2B: 4,
    CATEGORY_1B: 3,
    CATEGORY_C: 2,
}


class DevModel(Modeler):
    def __init__(
        self,
        league: str,
        season_start: str,
        season_end: str,
        ratings_type: str,
        category: str,
        role: str = None,
    ):
        self.league = league
        self.season_start = season_start
        self.season_end = season_end
        self.ratings_type = ratings_type
        self.file_mod = "-potential"
        self.category = category

        if self.category == CATEGORY_HITTING:
            self.target_predict_model = HittingModel(
                league, season_start, season_end, self.ratings_type, vsType="total"
            )
        elif self.category == CATEGORY_BASERUNNING:
            self.target_predict_model = BSRModel(
                league, season_start, season_end, self.ratings_type, "10"
            )
        elif self.category == CATEGORY_SP:
            self.target_predict_model = PitchingModel(
                league, season_start, season_end, "SP", self.ratings_type, False
            )
        elif self.category == CATEGORY_RP:
            self.target_predict_model = PitchingModel(
                league, season_start, season_end, "RP", self.ratings_type, False
            )
        elif self.category in [
            CATEGORY_C,
            CATEGORY_1B,
            CATEGORY_2B,
            CATEGORY_3B,
            CATEGORY_SS,
            CATEGORY_RF,
            CATEGORY_CF,
            CATEGORY_LF,
        ]:
            self.target_predict_model = FieldingModel(
                league,
                season_start,
                season_end,
                fielding_conversions[self.category],
                self.ratings_type,
            )
        self.target_predict_model.load_model()
        self.model = Modeler(feature_values[self.category], targets[self.category])

    def generate_prediction_data(self, data):

        predict_data = data.copy(deep=True)
        predict_data = predict_data[predict_data["Age"] == DEV_AGE_TARGET]

        if self.category == CATEGORY_HITTING:
            predict_data = predict_data[
                (predict_data["LPOS"] != "P")
                & (predict_data["LPOS"] != "SP")
                & (predict_data["LPOS"] != "RP")
                & (predict_data["LPOS"] != "CL")
            ]
            predict_data["lgwOBA"] = 0.31969
            predict_data["lgOBP"] = 0.31963
        elif self.category == CATEGORY_BASERUNNING:
            predict_data = predict_data[
                (predict_data["LPOS"] != "P")
                & (predict_data["LPOS"] != "SP")
                & (predict_data["LPOS"] != "RP")
                & (predict_data["LPOS"] != "CL")
            ]
        elif self.category == CATEGORY_RP or self.category == CATEGORY_SP:
            predict_data = predict_data[
                (predict_data["LPOS"] == "P")
                | (predict_data["LPOS"] == "SP")
                | (predict_data["LPOS"] == "RP")
                | (predict_data["LPOS"] == "CL")
            ]
            predict_data["RUNS_PER_OUT"] = 0.16948
            predict_data["lgwOBA"] = 0.31969
        elif self.category == CATEGORY_RF:
            predict_data = predict_data[predict_data["LPOS"] == "RF"]
        elif self.category == CATEGORY_CF:
            predict_data = predict_data[predict_data["LPOS"] == "CF"]
        elif self.category == CATEGORY_LF:
            predict_data = predict_data[predict_data["LPOS"] == "LF"]
        elif self.category == CATEGORY_SS:
            predict_data = predict_data[predict_data["LPOS"] == "SS"]
        elif self.category == CATEGORY_3B:
            predict_data = predict_data[predict_data["LPOS"] == "3B"]
        elif self.category == CATEGORY_2B:
            predict_data = predict_data[predict_data["LPOS"] == "2B"]
        elif self.category == CATEGORY_1B:
            predict_data = predict_data[predict_data["LPOS"] == "1B"]
        elif self.category == CATEGORY_C:
            predict_data = predict_data[predict_data["LPOS"] == "C"]

        predict_data[prediction_targets[self.category][0]] = 0

        return predict_data

    def filter_for_target_positions(self, data):
        # Filter out players who are not in the target positions
        if self.category == CATEGORY_HITTING:
            data = data[
                (data["LPOS"] != "P")
                & (data["LPOS"] != "SP")
                & (data["LPOS"] != "RP")
                & (data["LPOS"] != "CL")
            ]
        elif self.category == CATEGORY_BASERUNNING:
            data = data[
                (data["LPOS"] != "P")
                & (data["LPOS"] != "SP")
                & (data["LPOS"] != "RP")
                & (data["LPOS"] != "CL")
            ]
        elif self.category == CATEGORY_RP or self.category == CATEGORY_SP:
            data = data[
                (data["LPOS"] == "P")
                | (data["LPOS"] == "SP")
                | (data["LPOS"] == "RP")
                | (data["LPOS"] == "CL")
            ]
        elif self.category == CATEGORY_RF:
            data = data[data["RF"] > 0]
        elif self.category == CATEGORY_CF:
            data = data[data["CF"] > 0]
        elif self.category == CATEGORY_LF:
            data = data[data["LF"] > 0]
        elif self.category == CATEGORY_SS:
            data = data[data["SS"] > 0]
        elif self.category == CATEGORY_3B:
            data = data[data["3B"] > 0]
        elif self.category == CATEGORY_2B:
            data = data[data["2B"] > 0]
        elif self.category == CATEGORY_1B:
            data = data[data["1B"] > 0]
        elif self.category == CATEGORY_C:
            data = data[data["C"] > 0]

        return pd.DataFrame(data)

    def conform_data(self, data, generate_targets=True, max_age=DEV_AGE_TARGET):

        with pd.option_context("future.no_silent_downcasting", True):
            data.replace("-", 0, inplace=True, regex=False)

        # for col in feature_values[self.vsType]:
        #     if col not in exclude_adj:
        #         data[col] = data[col].apply(convert_80_rating

        # print(data.columns)
        # print (data['ID'].nunique())

        df_id = data["ID"]

        # Get Prediction on the target age for development.  The goal is to then
        # model the development of players from various ages to their target age.
        data = pd.DataFrame(data[data["Age"] <= max_age])

        if generate_targets:
            # print("PREDICTING TARGETS")
            predict_data = self.generate_prediction_data(data)
            predictions = self.target_predict_model.predict(
                self.season_start,
                0,
                skip_load=True,
                preloaded_data=pd.DataFrame(predict_data),
            )
            predictions.rename(
                columns={"Predictions": targets[self.category][0]}, inplace=True
            )
            master_data = data.merge(
                predictions[["ID", targets[self.category][0]]], on="ID", how="left"
            )
            master_data = master_data[master_data[targets[self.category][0]] > -500]
        else:
            # print("USING TARGETS")
            master_data = data
            master_data = self.filter_for_target_positions(master_data)
            master_data[targets[self.category][0]] = 0

            # if DEV_PREDICTION_AGE_MAX < DEV_AGE_TARGET:
            #    master_data = master_data[master_data["Age"] <= DEV_PREDICTION_AGE_MAX]

        filtered_data = master_data[
            feature_values[self.category] + targets[self.category]
        ]

        # create a dataset with a subset of the columns
        self.conform_column_types(
            filtered_data, feature_values[self.category] + targets[self.category]
        )

        return filtered_data, df_id

    def prepare_data(self, season, max_age=DEV_AGE_TARGET, generate_targets=True):
        # load fielding dataset from csv

        player_data = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-player-data.csv"
        )
        with pd.option_context("future.no_silent_downcasting", True):
            player_data.replace("-", 0, inplace=True)

        # combine fielding and player data
        return player_data

    def load_data(self):

        for season in range(int(self.season_start), int(self.season_end) + 1):
            filtered_data = self.prepare_data(season)

            self.filtered_data = (
                filtered_data
                if season == int(self.season_start)
                else pd.concat([self.filtered_data, filtered_data])
            )

        conformed_data, df_id = self.conform_data(
            self.filtered_data, generate_targets=True
        )

        # print(conformed_data.shape)
        # print(self.filtered_data.columns)
        # print (filtered_data['ID'].nunique())

        print(conformed_data.loc[conformed_data.isnull().any(axis=1)])

        self.model.load_data(conformed_data, targets[self.category][0])

    def load_single_season(self, season, max_age=DEV_AGE_TARGET):

        filtered_data = self.prepare_data(season, max_age=max_age)

        self.filtered_data = filtered_data

        conformed_data, df_id = self.conform_data(
            self.filtered_data, generate_targets=False
        )

        # print(conformed_data.shape)
        # print(self.filtered_data.columns)
        # print (filtered_data['ID'].nunique())

        return conformed_data, df_id

    def train(self, num_epochs: int):
        return self.model.train(num_epochs)

    def evaluate(self):
        return self.model.evaluate()

    def predict(
        self,
        season,
        skip_load=False,
        preloaded_data=None,
        generate_targets=False,
        max_age=DEV_AGE_TARGET,
    ):

        filtered_data, df_id = (
            self.conform_data(preloaded_data, generate_targets=generate_targets)
            if skip_load
            else self.load_single_season(season, max_age=max_age)
        )
        filtered_data = filtered_data.drop(columns=targets[self.category][0])
        results = self.model.predict(filtered_data)
        results["ID"] = df_id.copy()
        return results

    def feature_importance(self):
        return self.model.feature_importance()

    def save_model(self):
        self.model.save_model(
            f"./files/models/{self.ratings_type}-dev-{self.category}-model.pt"
        )

    def load_model(self):
        self.model.load_model(
            f"./files/models/{self.ratings_type}-dev-{self.category}-model.pt"
        )

    def load_released_model(self):
        self.model.load_model(
            f"./files/models/released/{self.ratings_type}-dev-{self.category}-model.pt"
        )
