import pandas as pd
from model import Modeler

feature_values = [
    "BABIP",
    "GAP",
    "POW",
    "EYE",
    "K's",
    # "BBT",
    "GBT",
    "FBT",
    "SPE",
    # "BUN",
    "BFH",
    # "B",
    # "WT",
    "Age",
    "RUN",
]

targets = ["wRAA600"]


class HittingModel(Modeler):
    def __init__(
        self, league: str, season_start: str, season_end: str, ratings_type: str
    ):
        self.league = league
        self.season_start = season_start
        self.season_end = season_end
        self.ratings_type = ratings_type
        self.model = Modeler(feature_values, targets)

    def load_data(self, pa_limit=100):

        for season in range(int(self.season_start), int(self.season_end) + 1):
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

            # combine fielding and player data
            master_data = hitting.merge(player_data, on="ID")
            master_data = master_data[master_data["PA"] >= pa_limit]
            filtered_data = master_data[feature_values + targets]

            # create a dataset with a subset of the columns
            self.conform_column_types(filtered_data, feature_values + targets)

            self.filtered_data = (
                filtered_data
                if season == int(self.season_start)
                else pd.concat([self.filtered_data, filtered_data])
            )
            print(self.filtered_data.shape)

        print(filtered_data.loc[filtered_data.isnull().any(axis=1)])

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
        self.model.save_model(f"./files/models/{self.ratings_type}-hitting-model.pt")

    def load_model(self):
        self.model.load_data(f"./files/models/{self.ratings_type}-hitting-model.pt")
