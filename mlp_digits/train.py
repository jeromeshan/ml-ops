import pickle

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def train(model_filename="model.pickle"):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(10, 10), random_state=1, max_iter=300
    )  # noqa: E501
    mlp.fit(X_train, y_train)

    pickle.dump(mlp, open(model_filename, "wb"))
    train_acc = accuracy_score(y_train, mlp.predict(X_train))
    test_acc = accuracy_score(y_test, mlp.predict(X_test))

    print("Train accuracy: " + str(train_acc))
    print("Test accuracy: " + str(test_acc))

    return train_acc, test_acc
