import os

import git
import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig, model_filename="../model.onnx"):
    os.system("cd .. && dvc pull data && cd mlp_digits")
    with open("../data/X_train.npy", "rb") as f:
        X_train = np.load(f)
    with open("../data/X_test.npy", "rb") as f:
        X_test = np.load(f)
    with open("../data/y_train.npy", "rb") as f:
        y_train = np.load(f)
    with open("../data/y_test.npy", "rb") as f:
        y_test = np.load(f)

    mlp = MLPClassifier(
        hidden_layer_sizes=(
            cfg.hyperparams.layer_size,
            cfg.hyperparams.layer_size,
        ),  # noqa: E501
        random_state=cfg.hyperparams.seed,
        max_iter=cfg.hyperparams.max_iter,
    ).fit(X_train, y_train)

    initial_type = [("int_input", Int64TensorType([None, 64]))]
    onx = convert_sklearn(mlp, initial_types=initial_type)
    with open(model_filename, "wb") as f:
        f.write(onx.SerializeToString())

    train_acc = accuracy_score(y_train, mlp.predict(X_train))
    test_acc = accuracy_score(y_test, mlp.predict(X_test))

    print("Train accuracy: " + str(train_acc))
    print("Test accuracy: " + str(test_acc))

    mlflow.set_tracking_uri(uri="http://128.0.1.1:8080")

    mlflow.set_experiment("mlp digits")
    with mlflow.start_run():

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        params = {
            "git commit id hash": str(sha),
            "hidden_layer_sizes": cfg.hyperparams.layer_size,
            "random_state": cfg.hyperparams.seed,
            "max_iter": cfg.hyperparams.max_iter,
        }

        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("Train accuracy", train_acc)
        mlflow.log_metric("Test accuracy", train_acc)
        for loss in mlp.loss_curve_:
            mlflow.log_metric("Loss", loss)

    return mlp


if __name__ == "__main__":
    train()
