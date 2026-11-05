import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shap
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util.device import get_device

from model.converters import *  # noqa: F401,F403,E402
from model.converters import ratings_conversions_80, ratings_conversions_10  # noqa: F401,E402


class MLPexplicit(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


class RegressionRunner:
    def __init__(self, feature_values: list[str]):
        self.num_features = len(feature_values)
        np.random.seed(42)
        self.device = get_device()
        self.model = MLPexplicit(self.num_features).to(self.device)

    def create_X_y(self, data: pd.DataFrame, targetCol: str):
        y = data[targetCol]
        X = data.drop(columns=[targetCol])

        return X, y

    def load_data(self, data: pd.DataFrame, targetCol: str):
        X, y = self.create_X_y(data, targetCol)

        # Check if the number of features matches the expected number of features
        if X.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, but got {X.shape[1]} features")
        X_scaled = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.50)

        # Convert the data to PyTorch tensors
        dev = self.device
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=dev)
        self.y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=dev).view(-1, 1)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=dev)
        self.y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32, device=dev).view(-1, 1)

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
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Check if the number of features matches the expected number of features
        if X.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, but got {X.shape[1]} features")

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu()

    def feature_importance(self, feature_values: list[str]):
        self.model.eval()
        # SHAP runs on CPU: copy the model and data off the accelerator
        cpu_model = copy.deepcopy(self.model).cpu()
        X_train_cpu = self.X_train_tensor.cpu()
        explainer = shap.GradientExplainer(cpu_model, X_train_cpu)

        # perm = torch.randperm(self.X_train_tensor.size(0))
        # k = 5
        # idx = perm[:k]
        samples = X_train_cpu
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
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, path)
        # for param_tensor in self.model.state_dict():
        #    print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        # print("Model's state_dict:")
        # for param_tensor in self.model.state_dict():
        #    print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())


class Modeler:
    def __init__(self, feature_values: list[str], targets: list[str]):
        self.feature_values = feature_values
        self.targets = targets
        self.model = RegressionRunner(feature_values)
        self.device = self.model.device

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
