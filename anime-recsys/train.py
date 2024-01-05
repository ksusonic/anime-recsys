import logging
import time

import hydra

# MLOps tools
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd

# Model Training
import tensorflow as tf
from callbacks import Callbacks
from model import recommender_net
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment("Predicting News Category Using NLP")

    # Load the dataset
    df = pd.read_csv(cfg["user_scores_dataset_path"], usecols=["user_id", "anime_id", "rating"])
    log.info("Shape of the Dataset: %s", df.shape)
    log.debug("Average Score: %f", np.mean(df["rating"]))

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

    # Printing dataset information
    log.debug("Number of unique users: %d, Number of unique anime: %d", num_users, num_animes)
    log.debug("Minimum rating: %d, Maximum rating: %d", min(df["rating"]), max(df["rating"]))

    # Shuffle the dataset
    df = shuffle(df, random_state=100)

    # Create feature matrix X and target variable y
    X = df[["user_encoded", "anime_encoded"]].values
    y = df["scaled_score"].values

    test_set_size = cfg["test_set_size"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=73)

    log.debug("Number of samples in the training set: %d", len(y_train))
    log.debug("Number of samples in the test set: %d", len(y_test))

    X_train_array = [X_train[:, 0], X_train[:, 1]]
    X_test_array = [X_test[:, 0], X_test[:, 1]]

    # Checking if TPU is initialized
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        log.debug("All devices: %s", tf.config.list_logical_devices("TPU"))
        tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)
        use_tpu = True
    except ValueError as e:
        log.warning("No TPU available: %s", e)
        use_tpu = False
        tpu_strategy = None

    # Checking if TPU is initialized and create the model accordingly
    if use_tpu:
        with tpu_strategy.scope():
            model = recommender_net(num_users, num_animes)
    else:
        model = recommender_net(num_users, num_animes)

    # Printing my model summary
    log.debug("Model summary: %s", model.summary())

    start_lr = cfg["start_lr"]
    min_lr = cfg["min_lr"]
    max_lr = cfg["max_lr"]
    batch_size = cfg["batch_size"]

    # Adjust the maximum learning rate and batch size if using TPU
    if use_tpu:
        log.debug("With TPU acceleration: max_lr and batch_size *= %d", tpu_strategy.num_replicas_in_sync)
        max_lr *= tpu_strategy.num_replicas_in_sync
        batch_size *= tpu_strategy.num_replicas_in_sync

    # Define the number of epochs for ramp-up, sustain, and exponential decay
    rampup_epochs = cfg["rampup_epochs"]
    sustain_epochs = cfg["sustain_epochs"]
    exp_decay = cfg["exp_decay"]

    callbacks = (
        Callbacks()
        # .with_checkpoints()
        .with_lr_scheduler(start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay).with_early_stopping()
    )

    # Define the list of callbacks
    epochs = cfg["epochs"]

    # Model training
    mlflow.keras.autolog()
    log.info("Starting learn...")
    start_time = time.time()

    model.fit(
        x=X_train_array,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test_array, y_test),
        callbacks=callbacks.to_list(),
    )
    end_time = time.time()
    log.info("Learn took: %d", end_time - start_time)

    if callbacks.checkpoints_dir is not None:
        model.load_weights(callbacks.checkpoints_dir)

    model_save_path = "models/resnet18.pt"
    mlflow.keras.save_model(model, model_save_path)
    log.info("Saved model at: %s", model_save_path)


if __name__ == "__main__":
    train()
