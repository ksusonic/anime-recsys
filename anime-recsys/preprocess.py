import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

log = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame, random_state: int) -> [pd.DataFrame, int, int]:
    # Preproces
    scaler = MinMaxScaler(feature_range=(0, 1))
    log.info("Using %s scaler", scaler.__class__.__name__)
    df["scaled_score"] = scaler.fit_transform(df[["rating"]])

    # Encoding user IDs
    user_encoder = LabelEncoder()
    df["user_encoded"] = user_encoder.fit_transform(df["user_id"])
    num_users = len(user_encoder.classes_)

    # Encoding anime IDs
    anime_encoder = LabelEncoder()
    df["anime_encoded"] = anime_encoder.fit_transform(df["anime_id"])
    num_animes = len(anime_encoder.classes_)

    return shuffle(df, random_state=random_state), num_users, num_animes
