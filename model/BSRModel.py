import pandas as pd
from model import Modeler

feature_values = [
    # "Age",
    # "CON",
    "SPE",
    "SR",
    "STE",
    "RUN",
    "WT",
    # "BABIP",
    # "GAP",
    # "K's",
    # "BFH",
    # "GBT",
    # "FBT",
]

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
    "WT",
]

targets = ["BSR600"]


class BSRModel(Modeler):
    def __init__(
        self,
        league: str,
        season_start: str,
        season_end: str,
        ratings_type: str,
        scale: str,
    ):
        self.league = league
        self.season_start = season_start
        self.season_end = season_end
        self.ratings_type = ratings_type
        self.scale = scale
        self.model = Modeler(feature_values, targets)

    def conform_data(self, master_data):
        df_id = master_data["ID"]
        filtered_data = master_data[feature_values + [targets[0]]]

        # create a dataset with a subset of the columns
        self.conform_column_types(filtered_data, feature_values + targets)

        return filtered_data, df_id

    def prepare_data(self, season, pa_limit):
        # load fielding dataset from csv

        hitting = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-hitting.csv"
        )
        player_data = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}-player-data.csv"
        )
        with pd.option_context("future.no_silent_downcasting", True):
            hitting.replace("-", 0, inplace=True)
            player_data.replace("-", 0, inplace=True)

        # for col in feature_values:
        #    if col not in exclude_adj:
        #        player_data[col] =
        # player_data[col].apply(convert_80_rating) if self.scale == "80" else player_data[col].apply(convert_10_rating)

        # combine fielding and player data
        master_data = hitting.merge(player_data, on="ID")
        master_data = master_data[master_data["PA"] >= pa_limit]

        return self.conform_data(master_data)

    def load_data(self, pa_limit=300):
        for season in range(int(self.season_start), int(self.season_end) + 1):
            filtered_data, df_id = self.prepare_data(season, pa_limit)

            self.filtered_data = (
                filtered_data
                if season == int(self.season_start)
                else pd.concat([self.filtered_data, filtered_data])
            )

        print(self.filtered_data.loc[self.filtered_data.isnull().any(axis=1)])

        print(self.filtered_data.shape)

        self.model.load_data(self.filtered_data, targets[0])

    def train(self, num_epochs: int):
        return self.model.train(num_epochs)

    def evaluate(self):
        return self.model.evaluate()

    def predict(self, season, pa_limit, skip_load=False, preloaded_data=None):
        filtered_data, df_id = (
            self.conform_data(preloaded_data)
            if skip_load
            else self.prepare_data(season, pa_limit)
        )
        filtered_data = filtered_data.drop(columns=targets[0])
        results = self.model.predict(filtered_data)
        results["ID"] = df_id.copy()
        return results

    def feature_importance(self):
        return self.model.feature_importance()

    def save_model(self):
        self.model.save_model(
            f"./files/models/{self.ratings_type}-baserunning-model.pt"
        )

    def load_model(self):
        self.model.load_model(
            f"./files/models/{self.ratings_type}-baserunning-model.pt"
        )

    def load_released_model(self):
        self.model.load_model(
            f"./files/models/released/{self.ratings_type}-baserunning-model.pt"
        )
