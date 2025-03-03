import pickle
import json

import pandas as pd

from mc.analyzers import Predictor


def load_file(path: str, type="pickle"):
    if type == "pickle":
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif type == "json":
        with open(path, "r") as f:
            data = json.load(f)
    elif type == "csv":
        data = pd.read_csv(path)
    else:
        raise ValueError("Type not supported")
    return data


def get_predictor() -> Predictor:
    dump = load_file("data/model.pkl", "pickle")[(0.7, 0.1, 0.2)]
    ranker = dump["ranker"]
    scaler = dump["scaler"]
    return Predictor(ranker, scaler)
