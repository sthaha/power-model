from typing import NamedTuple
from tabulate import tabulate
import pandas as pd
import json
import pathlib
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import typing
import logging

import numpy as np
import joblib

from power_model.datasource import prometheus


# TODO: delete me
import ipdb

logger = logging.getLogger(__name__)


def query_prometheus(prometheus_url, query, start_time, end_time):
    client = prometheus.Client(prometheus_url)
    # Execute the range query using the Client class and return the resulting DataFrame.
    return client.range_query(start=start_time, end=end_time, step="20s", **queries)


def save_to_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


class ErrorMetrics(NamedTuple):
    mae: float
    mse: float
    mape: float


def calculate_metrics(y_true, y_pred) -> ErrorMetrics:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE calculation
    return ErrorMetrics(mae, mse, mape)


def xgboost_model(X, y) -> tuple[XGBRegressor, ErrorMetrics]:
    xgb_model = XGBRegressor()
    xgb_model.fit(X, y)
    y_pred = xgb_model.predict(X)
    metrics = calculate_metrics(y, y_pred)

    # Save model with Joblib

    return xgb_model, metrics


Regressor = typing.TypeVar("Regressor")


def train_one(name: str, model: Regressor, X, y, model_path: pathlib.Path) -> tuple[Regressor, ErrorMetrics]:
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = calculate_metrics(y, y_pred)

    # Save model with Joblib
    joblib.dump(model, os.path.join(model_path, f"{name}_model.joblib"))
    save_to_json(metrics._asdict(), os.path.join(model_path, f"{name}_model_error.json"))

    return model, metrics


# TODO: fix typing.Any
def regressor_for_model_name(name: str, params: dict[str, typing.Any]) -> typing.Union[XGBRegressor, LinearRegression]:
    if name == "xgboost":
        return XGBRegressor(**params)
    if name == "linear":
        return LinearRegression(**params)
    if name == "logistic":
        return LogisticRegression(**params)

    raise ValueError(f"Invalid model name: {name}")


def train_models(X, y, models: dict[str, typing.Any], base_dir: pathlib.Path):
    trained_models = {}
    metrics = {}

    model_path = pathlib.Path(base_dir) / "models"
    os.makedirs(model_path, exist_ok=True)

    for name, params in models.items():
        regressor = regressor_for_model_name(name, params or {})
        model, err_metrics = train_one(name, regressor, X, y, model_path)
        trained_models[name] = model
        metrics[name] = err_metrics

    return trained_models, metrics


def train(config):
    prometheus_url = config["prometheus"]["url"]

    start_at: datetime = config["train"]["start_at"]
    end_at: datetime = config["train"]["end_at"]

    features = config["train"]["features"]
    learn_query = config["train"]["learn"]

    prom = prometheus.Client(prometheus_url)
    df_features = prom.range_query(start=start_at, end=end_at, **features)
    df_learn = prom.range_query(start=start_at, end=end_at, learn=learn_query)

    df = pd.merge(df_features, df_learn, on="timestamp")

    # Prepare training data (X and y)
    X = df[features.keys()].values
    y = df["learn"].values  # Adjust based on actual structure

    # save df
    model_base_path = pathlib.Path(config["train"]["path"])
    os.makedirs(model_base_path, exist_ok=True)
    df.to_csv(model_base_path / "training_input.csv")

    # Train models specified in the config
    trained_models, metrics = train_models(X, y, config["train"]["models"], model_base_path)
    # Save results to JSON file
    save_to_json({m: metrics[m]._asdict() for m in metrics.keys()}, model_base_path / "model_error.json")

    # Prepare data for tabulation
    table_data = []
    for model, metrics in metrics.items():
        table_data.append([model, metrics.mape, metrics.mae, metrics.mse])

    # Print the table
    print(tabulate(table_data, headers=["Name", "MAPE", "MAE", "MSE"], tablefmt="tabulate"))


class Predictor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        model_path = pathlib.Path(pipeline["train"]["path"]) / "models"
        models = pipeline["train"]["models"]
        self.models = {m: joblib.load(os.path.join(model_path, f"{m}_model.joblib")) for m in models}

        self.features = pipeline["train"]["features"]
        self.target = pipeline["train"]["learn"]

        self.prom = prometheus.Client(pipeline["prometheus"]["url"])

    def predict(self):
        now = datetime.now()

        df_features = self.prom.instant_query(at=now, **self.features)
        df_y = self.prom.instant_query(at=now, target=self.target)

        X = df_features[self.features.keys()].values
        y = df_y["target"].values

        table = []

        for name, model in self.models.items():
            y_pred = model.predict(X)
            metrics = calculate_metrics(y, y_pred)
            row = [name]
            row = np.append(row, X)
            row = np.append(row, y)
            row = np.append(row, y_pred)
            row = np.append(row, [metrics.mape, metrics.mae, metrics.mse])
            table.append(row.tolist())

        print(
            tabulate(
                table,
                headers=["Name", *self.features.keys(), "Target", "Predicted", "MAPE", "MAE", "MSE"],
                tablefmt="tabulate",
            )
        )


def create_predictor(pipeline) -> Predictor:
    return Predictor(pipeline)
