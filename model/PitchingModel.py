import pandas as pd
from model import (
    Modeler,
    convert_height_to_inches,
    convert_groundball_flyball,
    convert_velocity,
    convert_throws,
    convert_pitch_type,
    convert_slot,
)

feature_values = {
    "SP": [
        "HT",
        "WT",
        "T",
        "STU",
        "CON.1",
        "PBABIP",
        "HRR",
        "PIT",
        "G/F",
        "VELO",
        "Slot",
        "PT",
        "STM",
        "HLD",
    ],
    "RP": [
        "HT",
        "WT",
        "T",
        "STU",
        "CON.1",
        "PBABIP",
        "HRR",
        "PIT",
        "G/F",
        "VELO",
        "Slot",
        "PT",
        "STM",
        "HLD",
    ],
}

targets = {"SP": ["WAA200"], "RP": ["WAA200"]}


class PitchingModel(Modeler):
    def __init__(
        self,
        league: str,
        season_start: str,
        season_end: str,
        role: str,
        ratings_type: str,
    ):
        self.league = league
        self.season_start = season_start
        self.season_end = season_end
        self.role = role
        self.ratings_type = ratings_type
        self.model = Modeler(feature_values[self.role], targets[self.role])

    def load_data(self, ip_limit=100):

        for season in range(int(self.season_start), int(self.season_end) + 1):
            # load fielding dataset from csv
            pitching = pd.read_csv(
                f"./files/{self.league}/{season}/output/{self.league}-{season}-pitching.csv"
            )
            player_data = pd.read_csv(
                f"./files/{self.league}/{season}/output/{self.league}-{season}-player-data.csv"
            )
            player_data["WT"] = player_data["WT"].apply(lambda x: int(x[:3]))
            player_data["HT"] = player_data["HT"].apply(convert_height_to_inches)
            player_data["G/F"] = player_data["G/F"].apply(convert_groundball_flyball)
            player_data["VELO"] = player_data["VELO"].apply(convert_velocity)
            player_data["T"] = player_data["T"].apply(convert_throws)
            player_data["PT"] = player_data["PT"].apply(convert_pitch_type)
            player_data["Slot"] = player_data["Slot"].apply(convert_slot)
            with pd.option_context("future.no_silent_downcasting", True):
                pitching.replace("-", 0, inplace=True)
                player_data.replace("-", 0, inplace=True)

            # combine fielding and player data
            master_data = pitching.merge(player_data, on="ID")
            master_data = master_data[master_data["IPClean"] >= ip_limit]
            master_data = master_data[master_data["PRole"] >= self.role]
            filtered_data = master_data[feature_values[self.role] + targets[self.role]]

            # create a dataset with a subset of the columns
            self.conform_column_types(
                filtered_data, feature_values[self.role] + targets[self.role]
            )

            self.filtered_data = (
                filtered_data
                if season == int(self.season_start)
                else pd.concat([self.filtered_data, filtered_data])
            )
            print(self.filtered_data.shape)

        print(self.filtered_data.loc[self.filtered_data.isnull().any(axis=1)])

        self.model.load_data(self.filtered_data, targets[self.role][0])

    def train(self, num_epochs: int):
        return self.model.train(num_epochs)

    def evaluate(self):
        return self.model.evaluate()

    def predict(self, X):
        return self.model.predict_wrapper(X)

    def feature_importance(self):
        return self.model.feature_importance()

    def save_model(self):
        self.model.save_model(
            f"./files/models/{self.ratings_type}-pitching-{self.role}-model.pt"
        )

    def load_model(self):
        self.model.load_data(
            f"./files/models/{self.ratings_type}-pitching-{self.role}-model.pt"
        )
