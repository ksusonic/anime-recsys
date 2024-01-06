import logging

import hydra
import mlflow
import mlflow.keras
import pandas as pd
from omegaconf import DictConfig
from preprocess import preprocess
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def infer(cfg: DictConfig):
    random_state = cfg["random_state"]

    df = pd.read_csv(cfg["user_scores_dataset_path"], usecols=["user_id", "anime_id", "rating"])
    df, _, _ = preprocess(df, random_state)

    X = df[["user_encoded", "anime_encoded"]].values
    y = df["scaled_score"].values

    log.info("Splitting dataset by %f of df", cfg["inference_data_size"])
    _, X_test, _, y_test = train_test_split(X, y, test_size=cfg["inference_data_size"], random_state=random_state)

    model = mlflow.keras.load_model(cfg["model_save_path"])
    log.info("Loaded model: %s", model.summary())

    y_pred = model.predict([X_test[:, 0], X_test[:, 1]])

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    log.info("MAE: %f\nMSE: %f", mae, mse)

    df = pd.DataFrame(X_test, columns=['user_encoded', 'anime_encoded'])
    df['pred_score'] = y_pred

    pred_path = "outputs/pred.csv"
    df.to_csv(pred_path, index=False)
    log.info("Saved results to %s", pred_path)


if __name__ == "__main__":
    infer()
