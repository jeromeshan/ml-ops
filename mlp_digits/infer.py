import pickle

import numpy as np


def infer(model_filename="model.pickle", data=None):
    if data is None:
        data = np.random.randint(0, 17, size=(1, 64))
    loaded_model = pickle.load(open(model_filename, "rb"))
    return loaded_model.predict(data)[0]
