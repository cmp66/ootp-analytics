import pandas as pd
from model import Modeler

feature_values = {
    "SP": [
        # "T",
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
        # "VELO",
        # "STM",
        "HLD",
        "PIT",
        # "G/F",
        # "lgwOBA",
        # "lgOBP",
        "RUNS_PER_OUT",
        # "HT",
        # "Slot",
    ],
    "RP": [
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
        # "lgwOBA",
        # "lgOBP",
        # "HT",
        "RUNS_PER_OUT",
        # "T",
        # "G/F",
        # "VELO",
        # "Slot",
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

    def conform_data(self, data):

        df_id = data["ID"]
        filtered_data = data[feature_values[self.role] + [targets[self.role][0]]]

        # create a dataset with a subset of the columns
        self.conform_column_types(
            filtered_data, feature_values[self.role] + targets[self.role]
        )

        return filtered_data, df_id

    def prepare_data(self, season, ip_limit):
        # load fielding dataset from csv
        pitching = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-pitching.csv"
        )
        player_data = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-player-data.csv"
        )
        with pd.option_context("future.no_silent_downcasting", True):
            pitching.replace("-", 0, inplace=True)
            player_data.replace("-", 0, inplace=True)

        # combine fielding and player data
        master_data = pitching.merge(player_data, on="ID")
        master_data = master_data[master_data["IPClean"] >= ip_limit]
        master_data = master_data[master_data["PRole"] >= self.role]

        return self.conform_data(master_data)

    def load_data(self, ip_limit=100):

        for season in range(int(self.season_start), int(self.season_end) + 1):
            filtered_data, df_id = self.prepare_data(season, ip_limit)

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

    def predict(self, season, ip_limit, skip_load=False, preloaded_data=None):
        filtered_data, df_id = (
            self.conform_data(preloaded_data)
            if skip_load
            else self.prepare_data(season, ip_limit)
        )
        filtered_data = filtered_data.drop(columns=targets[self.role][0])
        results = self.model.predict(filtered_data)
        results["ID"] = df_id.copy()
        return results

    def feature_importance(self):
        return self.model.feature_importance()

    def save_model(self):
        self.model.save_model(
            f"./files/models/{self.ratings_type}-pitching-{self.role}-model.pt"
        )

    def load_model(self):
        self.model.load_model(
            f"./files/models/{self.ratings_type}-pitching-{self.role}-model.pt"
        )

    def load_released_model(self):
        self.model.load_model(
            f"./files/models/released/{self.ratings_type}-pitching-{self.role}-model.pt"
        )
