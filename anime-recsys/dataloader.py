import pathlib

import mlflow.keras
import pandas as pd
from dvc.api import DVCFileSystem


def load_user_scores(path: str):
    if not pathlib.Path(path).exists():
        DVCFileSystem().get("data", "data", recursive=True)
    return pd.read_csv(path, usecols=["user_id", "anime_id", "rating"])


def load_model(path: str):
    if not pathlib.Path(path).exists():
        DVCFileSystem().get("models", "models", recursive=True)
    return mlflow.keras.load_model(path)
