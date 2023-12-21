import os

import numpy as np
import onnxruntime as rt
import pandas as pd
from sklearn.metrics import accuracy_score


def infer(model_filename="../model.onnx"):
    os.system("cd .. && dvc pull data && cd mlp_digits")

    with open("../data/X_test.npy", "rb") as f:
        X_test = np.load(f).astype(np.int64)

    with open("../data/y_test.npy", "rb") as f:
        y_test = np.load(f)

    sess = rt.InferenceSession(
        model_filename, providers=["CPUExecutionProvider"]
    )  # noqa: E501
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred_onx = sess.run([label_name], {input_name: X_test})[0]
    acc = accuracy_score(y_test, pred_onx)
    pd.DataFrame(pred_onx).to_csv("../pred.csv")
    print("accuracy: ", acc)
    return acc


if __name__ == "__main__":
    infer()
