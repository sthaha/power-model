from typing import NamedTuple
from sklearn.preprocessing import PolynomialFeatures
from tabulate import tabulate
import pandas as pd
import json
import pathlib
import os
from datetime import datetime, timedelta

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import typing
import logging

import numpy as np
import joblib

from power_model.datasource import prometheus


# TODO: delete me
import ipdb

logger = logging.getLogger(__name__)


def save_to_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


class ErrorMetrics(NamedTuple):
    mae: float
    mse: float
    mape: float
    r2: float


def calculate_metrics(y_true, y_pred) -> ErrorMetrics:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = 0.0
    if len(y_true) > 5:
        r2 = r2_score(y_true, y_pred)

    assert type(r2) is float

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE calculation
    return ErrorMetrics(mae, mse, mape, r2)


Regressor = typing.TypeVar("Regressor")


def train_one(name: str, model: Regressor, X, y, model_path: pathlib.Path) -> tuple[Regressor, ErrorMetrics]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = calculate_metrics(y_val, y_pred)

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
    if name == "polynomial":
        poly = PolynomialFeatures(**params)
        return make_pipeline(poly, LinearRegression())

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


def create_model_for_feature(
    name: str,
    features: list[str],
    df: pd.DataFrame,
    train_path: pathlib.Path,
    models: dict[str, typing.Any],
):
    # Prepare training data (X and y)
    X = df[features].values
    y = df["target"].values

    model_base_path = train_path / name
    os.makedirs(model_base_path, exist_ok=True)
    df.to_csv(model_base_path / "training_inputs.csv")

    trained_models, metrics = train_models(X, y, models, model_base_path)
    # Save results to JSON file
    save_to_json({m: metrics[m]._asdict() for m in metrics.keys()}, model_base_path / "model_errors.json")

    table_data = []
    for model, metrics in metrics.items():
        table_data.append([model, metrics.mape, metrics.mae, metrics.mse, metrics.r2])

    # Print the table
    print(f"              {name}")
    print("----------------------------------")
    print(tabulate(table_data, headers=["Name", "MAPE", "MAE", "MSE", "R2"], tablefmt="tabulate"))


def train(config):
    prometheus_url = config["prometheus"]["url"]
    prom = prometheus.Client(prometheus_url)

    start_at: datetime = config["train"]["start_at"]
    end_at: datetime = config["train"]["end_at"]
    step = config["train"]["step"]

    target_query = config["train"]["target"]

    df_target = prom.range_query(start=start_at, end=end_at, step=step, target=target_query)

    groups = config["train"]["groups"]
    train_path = pathlib.Path(config["train"]["path"])

    for group in groups:
        name = group["name"]
        features = group["features"]
        df_features = prom.range_query(start=start_at, end=end_at, step=step, **features)
        df = pd.merge(df_features, df_target, on="timestamp")
        df.info()
        create_model_for_feature(name, features.keys(), df, train_path, config["train"]["models"])


class Predictor:
    def __init__(self, pipeline):
        self.pipeline = pipeline

        self.prom = prometheus.Client(pipeline["prometheus"]["url"])
        train = pipeline["train"]
        model_path = pathlib.Path(train["path"])

        self.groups = train["groups"]
        group_names = [g["name"] for g in self.groups]

        models = train["models"]
        self.models = {
            name: {m: joblib.load(model_path / name / "models" / f"{m}_model.joblib") for m in models}
            for name in group_names
        }

        self.target = train["target"]
        self.step = train["step"]

    def predict_range(self, start, end, step=None):
        step = step or self.step

        summary = []
        for group in self.groups:
            group_name = group["name"]
            features = group["features"]

            df_features = self.prom.range_query(start=start, end=end, step=step, **features)
            df_y = self.prom.range_query(start=start, end=end, step=step, target=self.target)

            X = df_features[features.keys()].values
            y = df_y["target"].values

            for model_name, model in self.models[group_name].items():
                y_pred = model.predict(X)
                metrics = calculate_metrics(y, y_pred)
                summary.append([group_name, model_name, metrics.mape, metrics.mae, metrics.mse, metrics.r2])
            summary.append([])

        print(
            tabulate(
                summary,
                headers=["Name", "MAPE", "MAE", "MSE", "R2"],
                tablefmt="tabulate",
            )
        )

    def predict(self, at=None):
        if at is None:
            at = datetime.now()

        df_y = self.prom.instant_query(at=at, target=self.target)
        y = df_y["target"].values

        for group in self.groups:
            group_name = group["name"]
            features = group["features"]
            df_features = self.prom.instant_query(at=at, **features)
            X = df_features[features.keys()].values

            table = []

            for model_name, model in self.models[group_name].items():
                y_pred = model.predict(X)
                diff = y - y_pred
                percent_error = np.round(abs(diff / y) * 100, 2)

                row = [group_name, model_name]
                row = np.append(row, X[0])
                row = np.append(row, y[0])
                row = np.append(row, y_pred[0])

                diff = y - y_pred
                percent_error = np.round(abs(diff / y) * 100, 2)
                row = np.append(row, [np.round(diff, 2), percent_error])
                table.append(row.tolist())

            print(
                tabulate(
                    table,
                    headers=["Group", "Name", *features.keys(), "Target", "Predicted", "Diff", "Err %"],
                    tablefmt="tabulate",
                )
            )


def create_predictor(pipeline) -> Predictor:
    return Predictor(pipeline)
