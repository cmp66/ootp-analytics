"""Model families and the training loop for the value models (plans/14-models.md).

Every family has the same interface: fit(X, y, w, groups) and predict(X), with X a DataFrame of the feature columns.
The scaler is part of the fitted model, so a prediction depends only on that row. The test data is never used for
choosing weights or stopping: the neural network stops on a validation set carved out of the training players."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from model import splits
from model.metrics import summarize
from util.device import get_device


def _frame(X, feature_names) -> np.ndarray:
    """The feature columns in training order as a float array (columns are matched by name, not position)."""
    if isinstance(X, pd.DataFrame):
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")
        values = X[feature_names].to_numpy(dtype=float)
    else:
        values = np.asarray(X, dtype=float)
        if values.shape[1] != len(feature_names):
            raise ValueError(f"expected {len(feature_names)} features, got {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError("features contain missing or infinite values")
    return values


class Scaler:
    """Mean / standard deviation of the training rows (weighted), applied to any later data."""

    def __init__(self, X: np.ndarray, w: np.ndarray | None = None):
        w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
        self.mean = np.average(X, axis=0, weights=w)
        spread = np.sqrt(np.average((X - self.mean) ** 2, axis=0, weights=w))
        self.scale = np.where(spread > 1e-12, spread, 1.0)  # a constant column stays 0 after scaling

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


class RidgeModel:
    family = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y, w, groups=None, seed: int = 0):
        self.feature_names = list(X.columns)
        values = _frame(X, self.feature_names)
        self.scaler = Scaler(values, w)
        self.model = Ridge(alpha=self.alpha).fit(self.scaler.transform(values), y, sample_weight=w)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self.scaler.transform(_frame(X, self.feature_names)))


class BoostingModel:
    family = "boosting"

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.05, max_leaf_nodes: int = 15, **extra):
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, **extra)

    def fit(self, X: pd.DataFrame, y, w, groups=None, seed: int = 0):
        self.feature_names = list(X.columns)
        values = _frame(X, self.feature_names)
        self.model = HistGradientBoostingRegressor(
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=seed, **self.params
        ).fit(values, y, sample_weight=w)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(_frame(X, self.feature_names))


@dataclass
class MLPConfig:
    hidden: tuple[int, ...] = (128, 64)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 25
    loss: str = "huber"  # "huber", "l1" or "mse"
    huber_delta: float = 1.0  # in units of the target's standard deviation
    val_fraction: float = 0.15  # of the training players, for stopping
    seed: int = 0
    device: str | None = None  # None: util.device.get_device()


def _network(n_features: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = n_features
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU()]
        width = size
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def _loss_fn(config: MLPConfig):
    if config.loss == "l1":
        return lambda pred, target: (pred - target).abs()
    if config.loss == "mse":
        return lambda pred, target: (pred - target) ** 2
    if config.loss == "huber":
        return nn.HuberLoss(reduction="none", delta=config.huber_delta)
    raise ValueError(f"unknown loss {config.loss!r}")


class MLPModel:
    family = "mlp"

    def __init__(self, config: MLPConfig | None = None):
        self.config = config or MLPConfig()

    def _device(self) -> torch.device:
        return torch.device(self.config.device) if self.config.device else get_device()

    def fit(self, X: pd.DataFrame, y, w, groups=None, seed: int | None = None):
        config = self.config
        seed = config.seed if seed is None else seed
        self.feature_names = list(X.columns)
        values = _frame(X, self.feature_names)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        groups = np.arange(len(y)) if groups is None else np.asarray(groups)

        # stopping uses players held out of the training rows, never the caller's test data
        train_idx, val_idx = splits.player_holdout(groups, config.val_fraction, seed)
        self.train_index, self.val_index = train_idx, val_idx  # stopping rows: their error is optimistic
        self.scaler = Scaler(values[train_idx], w[train_idx])
        self.y_mean = float(np.average(y[train_idx], weights=w[train_idx]))
        self.y_scale = float(np.sqrt(np.average((y[train_idx] - self.y_mean) ** 2, weights=w[train_idx]))) or 1.0

        device = self._device()
        torch.manual_seed(seed)

        def tensors(idx):
            xs = torch.tensor(self.scaler.transform(values[idx]), dtype=torch.float32, device=device)
            ys = torch.tensor((y[idx] - self.y_mean) / self.y_scale, dtype=torch.float32, device=device).view(-1, 1)
            ws = torch.tensor(w[idx] / w[idx].mean(), dtype=torch.float32, device=device).view(-1, 1)
            return xs, ys, ws

        x_train, y_train, w_train = tensors(train_idx)
        x_val, y_val, w_val = tensors(val_idx)
        net = _network(values.shape[1], config.hidden).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=max(2, config.patience // 4)
        )
        loss_fn = _loss_fn(config)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        batch = min(config.batch_size, max(16, len(train_idx) // 4))

        def weighted_loss(pred, target, weight):
            return (loss_fn(pred, target) * weight).sum() / weight.sum()

        best_loss, best_state, stale = float("inf"), None, 0
        self.history = []
        for epoch in range(config.max_epochs):
            net.train()
            order = torch.randperm(len(train_idx), generator=generator).to(device)
            for start in range(0, len(order), batch):
                stop = start + batch
                idx = order[start:stop]
                optimizer.zero_grad()
                weighted_loss(net(x_train[idx]), y_train[idx], w_train[idx]).backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                val_loss = float(weighted_loss(net(x_val), y_val, w_val))
            scheduler.step(val_loss)
            self.history.append(val_loss)
            if val_loss < best_loss - 1e-6:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= config.patience:
                    break
        net.load_state_dict(best_state)
        self.state = {k: v.cpu() for k, v in net.state_dict().items()}  # on the CPU whatever device trained it
        self.epochs_run = len(self.history)
        self.best_val_loss = best_loss
        return self

    def _network_on(self, device: torch.device) -> nn.Sequential:
        net = _network(len(self.feature_names), self.config.hidden)
        net.load_state_dict(self.state)
        return net.to(device).eval()

    def predict(self, X) -> np.ndarray:
        values = self.scaler.transform(_frame(X, self.feature_names))
        device = self._device()
        net = self._network_on(device)
        with torch.no_grad():
            out = net(torch.tensor(values, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        return out * self.y_scale + self.y_mean


FAMILIES = {"ridge": RidgeModel, "boosting": BoostingModel, "mlp": MLPModel}


def make_model(family: str, **params):
    if family not in FAMILIES:
        raise ValueError(f"unknown model family {family!r}; choose from {sorted(FAMILIES)}")
    return FAMILIES[family](**params) if family != "mlp" else MLPModel(MLPConfig(**params))


def fit_table(model, table, index=None, seed: int = 0):
    """Fit a model on (a subset of) a TrainingTable."""
    frame = table.frame if index is None else table.frame.iloc[index]
    return model.fit(frame[table.features], frame[table.target], frame[table.weight], frame[table.groups], seed=seed)


def cross_validate(family: str, table, n_splits: int = 5, seed: int = 0, **params):
    """Player-grouped k-fold: (out-of-fold predictions, metrics). Each player's rows are predicted by a model that
    never saw that player."""
    groups = table.frame[table.groups].to_numpy()
    prediction = np.full(len(table), np.nan)
    for train, test in splits.group_kfold(groups, n_splits, seed):
        fitted = fit_table(make_model(family, **params), table, train, seed)
        prediction[test] = fitted.predict(table.frame.iloc[test][table.features])
    return prediction, summarize(table.y, prediction, table.w)


def holdout_last_seasons(family: str, table, n_seasons: int = 5, seed: int = 0, **params):
    """Train on the early seasons and score the last n_seasons, as when predicting a season not yet seen."""
    train, test = splits.last_seasons_holdout(table.frame[table.season].to_numpy(), n_seasons)
    fitted = fit_table(make_model(family, **params), table, train, seed)
    prediction = fitted.predict(table.frame.iloc[test][table.features])
    frame = table.frame.iloc[test]
    return prediction, summarize(frame[table.target], prediction, frame[table.weight])
