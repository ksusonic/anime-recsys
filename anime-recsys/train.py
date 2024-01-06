import logging
import time

import hydra

# MLOps tools
import mlflow
import mlflow.keras
import numpy as np

# Model Training
import tensorflow as tf
from callbacks import Callbacks
from dataloader import load_user_scores
from model import recommender_net
from omegaconf import DictConfig, OmegaConf
from preprocess import preprocess
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    random_state = cfg["random_state"]

    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment("Anime Recommendation System")
    mlflow.log_params(dict(cfg))

    # Load the dataset
    df = load_user_scores(cfg["user_scores_dataset_path"])
    log.info("Shape of the Dataset: %s", df.shape)
    log.debug("Average Score: %f", np.mean(df["rating"]))

    df, num_users, num_animes = preprocess(df, random_state)
    # Printing dataset information
    log.debug("Number of unique users: %d, Number of unique anime: %d", num_users, num_animes)
    log.debug("Minimum rating: %d, Maximum rating: %d", min(df["rating"]), max(df["rating"]))

    # Create feature matrix X and target variable y
    X = df[["user_encoded", "anime_encoded"]].values
    y = df["scaled_score"].values

    test_set_size = cfg["test_set_size"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    log.debug("Number of samples in the training set: %d", len(y_train))
    log.debug("Number of samples in the test set: %d", len(y_test))

    X_train_array = [X_train[:, 0], X_train[:, 1]]
    X_val_array = [X_val[:, 0], X_val[:, 1]]
    # X_test_array = [X_test[:, 0], X_test[:, 1]]

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
        .with_checkpoints()
        .with_lr_scheduler(start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)
        .with_early_stopping()
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
        validation_data=(X_val_array, y_val),
        callbacks=callbacks.to_list(),
    )
    end_time = time.time()
    log.info("Learn took: %d", end_time - start_time)

    if callbacks.checkpoint is not None:
        model.load_weights(callbacks.checkpoint.name)

    model_save_path = cfg["model_save_path"]
    mlflow.keras.save_model(model, model_save_path)
    log.info("Saved model at: %s", model_save_path)


if __name__ == "__main__":
    train()
