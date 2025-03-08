import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shap
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    return int(height[0].strip()) * 12 + int(height[1].strip() * 30.48)


class MLP(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # self.fc1 = nn.Linear(num_features, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 16)
        # self.fc4 = nn.Linear(16, 1)
        # # self.fc5 = nn.Linear(32, 32)
        # self.fc6 = nn.Linear(32, 32)
        # self.fc7 = nn.Linear(32, 32)
        # self.fc8 = nn.Linear(32, 16)
        # self.fc9 = nn.Linear(16, 8)
        # self.fc10 = nn.Linear(8, 1)
        self.layers = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        # x = torch.relu(self.fc5(x))
        # x = torch.relu(self.fc6(x))
        # x = torch.relu(self.fc7(x))
        # x = torch.relu(self.fc8(x))
        # x = torch.relu(self.fc9(x))
        # x = self.fc4(x)
        return self.layers(x)


class RegressionRunner:
    def __init__(self, feature_values: list[str]):
        self.num_features = len(feature_values)
        self.model = MLP(self.num_features)
        # np.random.seed(42)

        # self.model =  nn.Sequential(
        #     nn.Linear(14, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64,32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)

    def load_data(self, data: pd.DataFrame, targetCol: str):
        y = data[targetCol]
        X = data.drop(columns=[targetCol])

        # Check if the number of features matches the expected number of features
        if X.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but got {X.shape[1]} features"
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, num_epochs: int):
        best_test = 10000.0
        best_eval = 10000.0

        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train_tensor)
            loss = self.criterion(outputs, self.y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if float(loss.item()) < best_test:
                best_weights = copy.deepcopy(self.model.state_dict())
                # eval_result = self.evaluate()
                best_test = float(loss.item())
                # if float(eval_result) < best_eval:
                # best_eval = float(eval_result)
                # print('Best Eval: ', best_eval)
                # best_weights = copy.deepcopy(self.model.state_dict())

                # else:
                #    retry += 1
                #    if retry > 2000:
                #        break

            if (epoch + 1) % 2000 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Best Loss: {best_test:.7f}, Best Eval: {best_eval:.7f}"
                )

            if loss.item() < 0.000999:
                print(f"Epoch [{epoch+1}/{num_epochs}], Best Loss: {best_test:.7f}")
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        return epoch, loss.item()

    def evaluate(self):
        self.model.eval()
        # with torch.no_grad():
        predictions = self.model(self.X_test_tensor)
        test_loss = self.criterion(predictions, self.y_test_tensor)
        # print(f"Test Absolute Error: {test_loss.item():.4f}")

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
