import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def convert_bbt(bbt: str) -> int:
    if bbt in ["Line Drive"]:
        return 1
    elif bbt in ["Flyball"]:
        return 3
    elif bbt in ["Normal"]:
        return 2
    elif bbt in ["Groundball"]:
        return 0


def convert_gbt(gbt: str) -> int:
    if gbt == "Ex. Pull":
        return 3
    elif gbt == "Pull":
        return 2
    elif gbt == "Normal":
        return 1
    elif gbt == "Spray":
        return 0


def convert_fbt(fbt: str) -> int:
    if fbt in ["Pull"]:
        return 2
    elif fbt in ["Normal"]:
        return 1
    elif fbt in ["Spray"]:
        return 0


def convert_bat_rl(rl: str) -> int:
    if rl == "R":
        return 1
    elif rl == "L":
        return


def convert_throws(rl: str) -> int:
    if rl == "R":
        return 1
    elif rl == "L":
        return 2


def convert_groundball_flyball(gbt: str) -> int:
    if gbt == "EX GB":
        return 1
    elif gbt == "GB":
        return 2
    elif gbt == "NEU":
        return 3
    elif gbt == "FB":
        return 4
    elif gbt == "EX FB":
        return 5


def convert_pitch_type(pt: str) -> int:
    if pt == "GB'er":
        return 1
    elif pt == "Normal":
        return 2
    elif pt == "Finesse":
        return 3
    elif pt == "Power":
        return 4


def convert_slot(slot: str) -> int:
    if slot == "OTT":
        return 1
    elif slot == "3/4":
        return 2
    elif slot == "SIDE":
        return 3
    elif slot == "SUB":
        return 4


def convert_velocity(vel: str) -> int:
    if vel == "100+":
        return 101
    velocity = vel.split("-")
    return (float(velocity[0].strip()) + float(velocity[1].strip())) / 2.0


def convert_height_to_inches(height: str) -> int:
    height = height.split("'")
    if len(height) != 3:
        raise ValueError("Height must be in the format 'X' Y'")
    return int(height[0].strip()) * 12 + int(height[1].strip())


class RegressionNN(nn.Module):
    def __init__(self, num_features):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class RegressionRunner:
    def __init__(self, feature_values: list[str]):
        self.num_features = len(feature_values)
        self.model = RegressionNN(self.num_features)

    def load_data(self, data: pd.DataFrame, targetCol: str):
        X = data.drop(columns=[targetCol])
        y = data[targetCol]

        # Check if the number of features matches the expected number of features
        if X.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but got {X.shape[1]} features"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert the data to PyTorch tensors
        self.X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(
            -1, 1
        )
        self.X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(
            -1, 1
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train_tensor)
            loss = self.criterion(outputs, self.y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 2000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}")

            if loss.item() < 0.000999:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}")
                break
        return epoch, loss.item()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_test_tensor)
            test_loss = self.criterion(predictions, self.y_test_tensor)
            # print(f"Test Mean Squared Error: {test_loss.item():.4f}")

        return test_loss.item()

    def predict_wrapper(self, X):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            return self.model(torch.Tensor(X)).numpy()

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

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


class Modeler:
    def __init__(self, feature_values: list[str], targets: list[str]):
        self.feature_values = feature_values
        self.targets = targets
        self.model = RegressionRunner(feature_values)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA is available! Using GPU.")
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")

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
        return self.model.predict_wrapper(X)

    def feature_importance(self):
        return self.model.feature_importance(self.feature_values)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
