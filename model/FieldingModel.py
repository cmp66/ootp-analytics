import pandas as pd
from model import Modeler, convert_height_to_inches


feature_values = {
    9: ["Age", "WT", "SPE", "OFRngDelta", "OFArmDelta", "OFERRDelta", "RF", "ID"],
    8: ["Age", "WT", "SPE", "OFRngDelta", "OFArmDelta", "OFERRDelta", "CF"],
    7: ["Age", "WT", "SPE", "OFRngDelta", "OFArmDelta", "OFERRDelta", "LF"],
    6: [
        "Age",
        "WT",
        "SPE",
        "IFRngDelta",
        "IFArmDelta",
        "IFTDPDelta",
        "IFErrDelta",
        "SS",
    ],
    5: [
        "Age",
        "WT",
        "SPE",
        "IFRngDelta",
        "IFArmDelta",
        "IFTDPDelta",
        "IFErrDelta",
        "3B",
    ],
    4: [
        "Age",
        "WT",
        "SPE",
        "IFRngDelta",
        "IFArmDelta",
        "IFTDPDelta",
        "IFErrDelta",
        "2B",
    ],
    3: [
        "Age",
        "WT",
        "SPE",
        "IFRngDelta",
        "IFArmDelta",
        "IFTDPDelta",
        "IFErrDelta",
        "HT",
        "1B",
    ],
    2: ["Age", "WT", "SPE", "CBLKDelta", "CARMDelta", "CFRMDelta", "C"],
}

targets = {
    9: ["runsPAdjSeason"],
    8: ["runsPAdjSeason"],
    7: ["runsPAdjSeason"],
    6: ["runsPAdjSeason"],
    5: ["runsPAdjSeason"],
    4: ["runsPAdjSeason"],
    3: ["runsPAdjSeason"],
    2: ["runsPAdjSeason"],
}


class FieldingModel(Modeler):
    def __init__(
        self,
        league: str,
        season_start: str,
        season_end: str,
        position: int,
        ratings_type: str,
    ):
        self.league = league
        self.season_start = season_start
        self.season_end = season_end
        self.position = position
        self.ratings_type = ratings_type
        self.model = Modeler(feature_values[self.position], targets)

    def load_data(self, position: int, ip_limit=200):

        for season in range(int(self.season_start), int(self.season_end) + 1):

            fielding = pd.read_csv(
                f"./files/{self.league}/{season}/output/{self.league}-{season}-fielding.csv"
            )
            player_data = pd.read_csv(
                f"./files/{self.league}/{season}/output/{self.league}-{season}-player-data.csv"
            )
            player_data["WT"] = player_data["WT"].apply(lambda x: int(x[:3]))
            player_data["HT"] = player_data["HT"].apply(convert_height_to_inches)
            with pd.option_context("future.no_silent_downcasting", True):
                fielding.replace("-", 0, inplace=True)
                player_data.replace("-", 0, inplace=True)

            # combine fielding and player data
            filtered_data = fielding.merge(player_data, on="ID")
            filtered_data = filtered_data[filtered_data["IPClean"] >= ip_limit]
            filtered_data = filtered_data[filtered_data["POS"] == self.position]
            filtered_data = filtered_data[
                feature_values[self.position] + targets[self.position]
            ]

            # create a dataset with a subset of the columns
            self.conform_column_types(
                filtered_data, feature_values[position] + targets[position]
            )
            self.filtered_data = (
                filtered_data
                if season == int(self.season_start)
                else pd.concat([self.filtered_data, filtered_data])
            )
            print(self.filtered_data.shape)

        print(self.filtered_data.loc[self.filtered_data.isnull().any(axis=1)])

        self.model.load_data(self.filtered_data, targets[position][0])

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
            f"./files/models/{self.ratings_type}-fielding-{self.position}-model.pt"
        )

    def load_model(self):
        self.model.load_data(
            f"./files/models/{self.ratings_type}-fielding-{self.position}-model.pt"
        )
