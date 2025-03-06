import pandas as pd
from model import Modeler, convert_bbt, convert_gbt, convert_fbt

feature_values = [
    "Age",
    "WT",
    "SPE",
    "SR",
    "STE",
    "RUN",
    "BABIP",
    "GAP",
    "K's",
    "BFH",
    "BBT",
    "GBT",
    "FBT",
]

targets = ["BSR600"]


class BSRModel(Modeler):
    def __init__(
        self, league: str, season_start: str, season_end: str, ratings_type: str
    ):
        self.league = league
        self.season_start = season_start
        self.season_end = season_end
        self.ratings_type = ratings_type
        self.model = Modeler(feature_values, targets)

    def load_data(self, pa_limit=100):
        # load fielding dataset from csv
        hitting = pd.read_csv(
            f"./files/{self.league}/{self.season_start}/output/{self.league}-{self.season_start}-hitting.csv"
        )
        player_data = pd.read_csv(
            f"./files/{self.league}/{self.season_start}/output/{self.league}-{self.season_start}-player-data.csv"
        )
        player_data["WT"] = player_data["WT"].apply(lambda x: int(x[:3]))
        player_data["BBT"] = player_data["BBT"].apply(convert_bbt)
        player_data["GBT"] = player_data["GBT"].apply(convert_gbt)
        player_data["FBT"] = player_data["FBT"].apply(convert_fbt)
        with pd.option_context("future.no_silent_downcasting", True):
            hitting.replace("-", 0, inplace=True)
            player_data.replace("-", 0, inplace=True)

        # combine fielding and player data
        master_data = hitting.merge(player_data, on="ID")
        master_data = master_data[master_data["PA"] >= pa_limit]
        self.filtered_data = master_data[feature_values + targets]

        # create a dataset with a subset of the columns
        self.conform_column_types(self.filtered_data, feature_values + targets)

        self.model.load_data(self.filtered_data, targets[0])

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
            f"./files/models/{self.ratings_type}-baserunning-model.pt"
        )

    def load_model(self):
        self.model.load_data(f"./files/models/{self.ratings_type}-baserunning-model.pt")
