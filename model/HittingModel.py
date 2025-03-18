import pandas as pd
from model import Modeler

conversion_to_potential = {
    "BABIP": "BA P",
    "CON": "CON P",
    "GAP": "GAP P",
    "POW": "POW P",
    "EYE": "EYE P",
    "K's": "K P",
    "BA vR": "BA vR",
    "BA vL": "BA vL",
    "GAP vR": "GAP vR",
    "GAP vL": "GAP vL",
    "POW vR": "POW vR",
    "POW vL": "POW vL",
    "EYE vR": "EYE vR",
    "EYE vL": "EYE vL",
    "K vR": "K vR",
    "K vL": "K vL",
    "lgwOBA": "lgwOBA",
    "lgOBP": "lgOBP",
}


feature_values = {
    "total": [
        # "ID",
        "BABIP",
        "BA vR",
        "BA vL",
        "GAP vR",
        "GAP vL",
        "POW",
        "POW vR",
        "POW vL",
        "EYE",
        "EYE vR",
        "EYE vL",
        "K's",
        "K vR",
        "K vL",
        "lgwOBA",
        "lgOBP",
        # "B",
        #########################
        # "GAP",
        # "lgK_RATE",
        # "wOBA_SCALE",
        # "BBT",
        # "GBT",
        # "FBT",
        # "SPE",
        # "BUN",
        # "BFH",
        # "HT",
        # "Age",
        # "RUN",
        # "lgXBH_RATE",
        # "lgBABIP",
        # "lgHR_RATE",
    ],
    "right": [
        # "ID",
        # "BABIP",
        "BA vR",
        # "BA vL",
        "GAP vR",
        # "GAP vL",
        # "POW",
        "POW vR",
        "POW vL",
        # "EYE",
        "EYE vR",
        # "EYE vL",
        "K's",
        "K vR",
        # "K vL",
        "lgwOBA",
        "lgOBP",
        # "B",
        #########################
        # "GAP",
        # "lgK_RATE",
        # "wOBA_SCALE",
        # "BBT",
        # "GBT",
        # "FBT",
        # "SPE",
        # "BUN",
        # "BFH",
        # "HT",
        # "Age",
        # "RUN",
        # "lgXBH_RATE",
        # "lgBABIP",
        # "lgHR_RATE",
    ],
    "left": [
        # "ID",
        # "BABIP",
        # "BA vR",
        "BA vL",
        # "GAP vR",
        "GAP vL",
        # "POW",
        # "POW vR",
        "POW vL",
        # "EYE",
        # "EYE vR",
        "EYE vL",
        # "K's",
        # "K vR",
        "K vL",
        "lgwOBA",
        "lgOBP",
        # "B",
        #########################
        # "GAP",
        # "lgK_RATE",
        # "wOBA_SCALE",
        # "BBT",
        # "GBT",
        # "FBT",
        # "SPE",
        # "BUN",
        # "BFH",
        # "HT",
        # "Age",
        # "RUN",
        # "lgXBH_RATE",
        # "lgBABIP",
        # "lgHR_RATE",
    ],
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
    "total": ["wRAA600"],
    "right": ["wRAA600Right"],
    "left": ["wRAA600Left"],
}


class HittingModel(Modeler):
    def __init__(
        self,
        league: str,
        season_start: str,
        season_end: str,
        ratings_type: str,
        vsType: str,
        use_potential: bool,
    ):
        self.league = league
        self.season_start = season_start
        self.season_end = season_end
        self.ratings_type = ratings_type
        self.vsType = vsType
        self.use_potential = use_potential
        self.file_mod = (
            ""
            if self.vsType == "total"
            else "-right" if self.vsType == "right" else "-left"
        )
        self.model = Modeler(feature_values[self.vsType], [targets[self.vsType][0]])

    def conform_data(self, data):

        with pd.option_context("future.no_silent_downcasting", True):
            data.replace("-", 0, inplace=True)

        # for col in feature_values[self.vsType]:
        #     if col not in exclude_adj:
        #         data[col] = data[col].apply(convert_80_rating

        df_id = data["ID"]

        filtered_data = data[feature_values[self.vsType] + [targets[self.vsType][0]]]

        # create a dataset with a subset of the columns
        self.conform_column_types(
            filtered_data, feature_values[self.vsType] + targets[self.vsType]
        )

        return filtered_data, df_id

    def prepare_data(self, season, pa_limit):
        # load fielding dataset from csv

        hitting = pd.read_csv(
            f"./files/{self.league}/{season}/output/{self.league}-{season}{self.file_mod}-hitting.csv"
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

        if self.use_potential:
            for k, v in conversion_to_potential.items():
                master_data[k] = master_data[v]

        return self.conform_data(master_data)

    def load_data(self, pa_limit=100):

        for season in range(int(self.season_start), int(self.season_end) + 1):
            filtered_data, df_id = self.prepare_data(season, pa_limit)

            self.filtered_data = (
                filtered_data
                if season == int(self.season_start)
                else pd.concat([self.filtered_data, filtered_data])
            )

        print(self.filtered_data.shape)

        print(filtered_data.loc[filtered_data.isnull().any(axis=1)])

        self.model.load_data(self.filtered_data, targets[self.vsType][0])

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
        filtered_data = filtered_data.drop(columns=targets[self.vsType][0])
        results = self.model.predict(filtered_data)
        results["ID"] = df_id.copy()
        return results

    def feature_importance(self):
        return self.model.feature_importance()

    def save_model(self):
        self.model.save_model(
            f"./files/models/{self.ratings_type}{self.file_mod}-hitting-model.pt"
        )

    def load_model(self):
        self.model.load_model(
            f"./files/models/{self.ratings_type}{self.file_mod}-hitting-model.pt"
        )

    def load_released_model(self):
        self.model.load_model(
            f"./files/models/released/{self.ratings_type}{self.file_mod}-hitting-model.pt"
        )
