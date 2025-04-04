import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shap
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ratings_conversions_80 = {
    -50: -1.000,
    -30: -0.315,
    -25: -0.155,
    -20: -0.75,
    -15: -0.35,
    -10: -0.15,
    -5: -0.05,
    0: 0,
    5: 0.05,
    10: 0.15,
    15: 0.35,
    20: 0.75,
    25: 0.155,
    30: 0.315,
}

ratings_conversions_10 = {
    -5: -155,
    -4: -75,
    -3: -35,
    -2: -15,
    -1: -5,
    0: 0,
    1: 5,
    2: 15,
    3: 35,
    4: 75,
    5: 155,
}


def convert_80_rating(rating: int) -> int:
    return float(rating) / 100.0


def convert_10_rating(rating: int) -> int:
    return float(rating) / 10.0


def convert_bbt(bbt: str) -> int:
    if bbt in ["Line Drive"]:
        return 0
    elif bbt in ["Flyball"]:
        return 0
    elif bbt in ["Normal"]:
        return 0
    elif bbt in ["Groundball"]:
        return 0


def convert_gbt(gbt: str) -> int:
    if gbt == "Ex. Pull":
        return 3
    elif gbt == "Pull":
        return 2
    elif gbt == "Normal":
        return 0
    elif gbt == "Spray":
        return 1


def convert_fbt(fbt: str) -> int:
    if fbt in ["Pull"]:
        return 2
    elif fbt in ["Normal"]:
        return 0
    elif fbt in ["Spray"]:
        return 1


def convert_bat_rl(rl: str) -> int:
    if rl == "R":
        return 1
    elif rl == "L":
        return 2
    elif rl == "S":
        return 3


def convert_throws(rl: str) -> int:
    if rl == "R":
        return 1
    elif rl == "L":
        return 2


def convert_groundball_flyball(gbt: str) -> int:
    if gbt == "EX GB":
        return 62
    elif gbt == "GB":
        return 60
    elif gbt == "NEU":
        return 50
    elif gbt == "FB":
        return 46
    elif gbt == "EX FB":
        return 42


def convert_pitch_type(pt: str) -> int:
    if pt == "GB'er":
        return 10
    elif pt == "Normal":
        return 20
    elif pt == "Finesse":
        return 30
    elif pt == "Power":
        return 40


def convert_slot(slot: str) -> int:
    if slot == "OTT":
        return 4
    elif slot == "3/4":
        return 3
    elif slot == "SIDE":
        return 2
    elif slot == "SUB":
        return 1


def convert_velocity(vel: str) -> int:
    if vel == "80-83":
        return 2
    elif vel == "83-85":
        return 3
    elif vel == "84-86":
        return 4
    elif vel == "85-87":
        return 5
    elif vel == "86-88":
        return 6
    elif vel == "87-89":
        return 7
    elif vel == "88-90":
        return 8
    elif vel == "89-91":
        return 9
    elif vel == "90-92":
        return 10
    elif vel == "91-93":
        return 11
    elif vel == "92-94":
        return 12
    elif vel == "93-95":
        return 13
    elif vel == "94-96":
        return 14
    elif vel == "95-97":
        return 15
    elif vel == "96-98":
        return 16
    elif vel == "97-99":
        return 17
    elif vel == "98-100":
        return 18
    elif vel == "99-101":
        return 19
    elif vel == "100+":
        return 20


def convert_height_to_cm(height: str) -> int:
    height = height.split("'")
    if len(height) != 3:
        raise ValueError("Height must be in the format 'X' Y'")
    return (float(height[0]) * 12.0 + float(height[1])) * 2.54


class MLPexplicit(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


class RegressionRunner:
    def __init__(self, feature_values: list[str]):
        self.num_features = len(feature_values)
        np.random.seed(42)
        self.model = MLPexplicit(self.num_features)

    def create_X_y(self, data: pd.DataFrame, targetCol: str):
        y = data[targetCol]
        X = data.drop(columns=[targetCol])

        return X, y

    def load_data(self, data: pd.DataFrame, targetCol: str):
        X, y = self.create_X_y(data, targetCol)

        # Check if the number of features matches the expected number of features
        if X.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but got {X.shape[1]} features"
            )
        X_scaled = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15)

        # Convert the data to PyTorch tensors
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(
            -1, 1
        )
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(
            -1, 1
        )

        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, num_epochs: int):
        best_test = 10000.0
        best_eval = 10000.0

        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train_tensor)
            loss = self.criterion(outputs, self.y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if float(loss.item()) < best_test:
                # best_weights = copy.deepcopy(self.model.state_dict())
                eval_result = self.evaluate()
                best_test = float(loss.item())
                if float(eval_result) < best_eval:
                    best_eval = float(eval_result)
                    # print('Best Eval: ', best_eval)
                    best_weights = copy.deepcopy(self.model.state_dict())

                # else:
                #    retry += 1
                #    if retry > 2000:
                #        break

            if (epoch + 1) % 2000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Best Loss: {best_test:.7f}")

            if loss.item() < 0.000999:
                print(f"Epoch [{epoch+1}/{num_epochs}], Best Loss: {best_test:.7f}")
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        return epoch, loss.item()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_test_tensor)
            test_loss = self.criterion(predictions, self.y_test_tensor)
        # print(f"Test Absolute Error: {test_loss.item():.4f}")

        return test_loss.item()

    def predict_wrapper(self, data: pd.DataFrame):

        scaler = StandardScaler()
        X = scaler.fit_transform(data)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Check if the number of features matches the expected number of features
        if X.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but got {X.shape[1]} features"
            )

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions

    def feature_importance(self, feature_values: list[str]):
        self.model.eval()
        explainer = shap.GradientExplainer(self.model, self.X_train_tensor)

        # perm = torch.randperm(self.X_train_tensor.size(0))
        # k = 5
        # idx = perm[:k]
        samples = self.X_train_tensor
        shap_values = explainer.shap_values(samples)

        df = pd.DataFrame(
            {
                "mean_abs_shap": np.mean(np.abs(shap_values), axis=0).reshape(-1),
                "stdev_abs_shap": np.std(np.abs(shap_values), axis=0).reshape(-1),
                "name": feature_values,
            }
        )
        return df.sort_values("mean_abs_shap", ascending=False)[: self.num_features]

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        # for param_tensor in self.model.state_dict():
        #    print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        # print("Model's state_dict:")
        # for param_tensor in self.model.state_dict():
        #    print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())


class Modeler:
    def __init__(self, feature_values: list[str], targets: list[str]):
        self.feature_values = feature_values
        self.targets = targets
        self.model = RegressionRunner(feature_values)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def load_data(self, data: pd.DataFrame, targetCol: str):
        self.model.load_data(data, targetCol)

    def conform_column_types(self, data: pd.DataFrame, columns_to_conform: list[str]):
        for column in columns_to_conform:
            data = data.astype({column: "float"})

        return data

    def train(self, num_epochs: int):
        return self.model.train(num_epochs)

    def evaluate(self):
        return self.model.evaluate()

    def predict(self, X):
        y_tensor = self.model.predict_wrapper(X)

        result_df = X.copy()
        result_df["Predictions"] = y_tensor.numpy()

        return result_df

    def feature_importance(self):
        return self.model.feature_importance(self.feature_values)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
