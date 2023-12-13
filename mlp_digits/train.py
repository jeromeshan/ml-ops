import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig, model_filename="model.onnx"):
    with open("data/X_train.npy", "rb") as f:
        X_train = np.load(f)
    with open("data/X_test.npy", "rb") as f:
        X_test = np.load(f)
    with open("data/y_train.npy", "rb") as f:
        y_train = np.load(f)
    with open("data/y_test.npy", "rb") as f:
        y_test = np.load(f)

    mlp = MLPClassifier(
        hidden_layer_sizes=(cfg.hyperparams.layer_size, cfg.hyperparams.layer_size), random_state=cfg.hyperparams.seed, max_iter=cfg.hyperparams.max_iter
    ).fit(  # noqa: E501
        X_train, y_train
    )

    initial_type = [("int_input", Int64TensorType([None, 64]))]
    onx = convert_sklearn(mlp, initial_types=initial_type)
    with open(model_filename, "wb") as f:
        f.write(onx.SerializeToString())

    train_acc = accuracy_score(y_train, mlp.predict(X_train))
    test_acc = accuracy_score(y_test, mlp.predict(X_test))

    print("Train accuracy: " + str(train_acc))
    print("Test accuracy: " + str(test_acc))

    return train_acc, test_acc


if __name__ == "__main__":
    train()