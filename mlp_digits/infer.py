import numpy as np
import onnxruntime as rt


def infer(model_filename="model.onnx", data=None):
    if data is None:
        data = np.random.randint(0, 17, size=(1, 64))
    data = data.astype(np.int64)
    sess = rt.InferenceSession(
        model_filename, providers=["CPUExecutionProvider"]
    )  # noqa: E501
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred_onx = sess.run([label_name], {input_name: data})[0][0]
    return pred_onx

if __name__ == "__main__":
    infer()