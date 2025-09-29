import numpy as np

# classifiers
from river.neighbors import KNNClassifier
from river.tree import HoeffdingTreeClassifier
from river.linear_model import LogisticRegression
from river.naive_bayes import GaussianNB

# preprocessing
from river.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    StandardScaler,
)

clf_config_online = {
    LogisticRegression: {
        "l2": [.0,.01,.001],
    },
    GaussianNB: {},
    HoeffdingTreeClassifier: {
        "max_depth": [10, 30, 60, 10, 30, 60],
        "grace_period": [10, 100, 200, 10, 100, 200],
        "max_size": [5, 10],
    },
    KNNClassifier: {
        "n_neighbors": [1, 5, 20],
        "window_size": [100, 500, 1000],
        "weighted": [True, False],
    },
    MinMaxScaler: {},
    StandardScaler: {},
    MaxAbsScaler: {},
}
