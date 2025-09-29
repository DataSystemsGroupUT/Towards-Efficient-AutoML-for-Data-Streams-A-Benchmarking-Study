from river import linear_model, naive_bayes, tree, neighbors, ensemble
from river import preprocessing
from river import feature_selection
from river import stats
from .helper import range_gen

# Define a list of model options to be used in the automated pipeline
model_options = [
    linear_model.Perceptron(),
    linear_model.LogisticRegression(),
    naive_bayes.GaussianNB(),
    tree.HoeffdingTreeClassifier(),
    ensemble.AdaptiveRandomForestClassifier(),
    neighbors.KNNClassifier(),
]

# Define a list of preprocessing options to be used in the automated pipeline
preprocessor_options = [preprocessing.MinMaxScaler(), preprocessing.StandardScaler()]

# Define a list of feature selection options to be used in the automated pipeline
feature_selection_options = [
    feature_selection.PoissonInclusion(p=0.1, seed=42),
    feature_selection.VarianceThreshold(threshold=0),
    feature_selection.SelectKBest(similarity=stats.PearsonCorr()),
]

# Define a dictionary of hyperparameter options for each model, preprocessor, and feature selector
hyperparameters_options = {
    "Perceptron": {
        "l2": range_gen(0.00, 0.01, step=0.001, float_n=True),
    },
    "LogisticRegression": {
        "l2": range_gen(0.00, 0.01, step=0.001, float_n=True),
    },
    "GaussianNB": {},
    "HoeffdingTreeClassifier": {
        "max_depth": range_gen(10, 100, step=10),
        "grace_period": range_gen(50, 500, step=50),
        "split_confidence": [1e-9, 1e-7, 1e-4, 1e-2],
        "tie_threshold": range_gen(0.02, 0.08, step=0.01, float_n=True),
        "nb_threshold": range_gen(0, 50, step=10),
        "split_criterion": ["info_gain", "gini", "hellinger"],
        "leaf_prediction": ["mc", "nb", "nba"],
    },
    "AdaptiveRandomForestClassifier": {
        "n_models": range_gen(3,9, step=1),
        "max_depth": range_gen(10, 100, step=10),
        "grace_period": range_gen(50, 500, step=50),
        "lambda_value": range_gen(2, 10, step=1),
        "split_confidence": range_gen(0.01, 0.1, step=0.01, float_n=True),
        "tie_threshold": range_gen(0.02, 0.08, step=0.01, float_n=True),
        "nb_threshold": range_gen(0, 50, step=10),
        "split_criterion": ["info_gain", "gini", "hellinger"],
        "leaf_prediction": ["mc", "nb", "nba"],
    },
    "KNNClassifier": {
        "n_neighbors": range_gen(3, 9, step=1),
        "window_size": range_gen(100, 5100, step=200),
        "weighted": [True, False],
        "p": range_gen(1, 5, step=1),
    },
    "MinMaxScaler": {},
    "StandardScaler": {"with_std": [True, False]},
    "PoissonInclusion": {"p": range_gen(0.1, 1.0, step=0.1, float_n=True)},
    "VarianceThreshold": {
        "threshold": range_gen(0.0, 1.0, step=0.1, float_n=True),
        "min_samples": range_gen(1, 10, step=1),
    },
    "SelectKBest": {
        "k": range_gen(1, 25, step=1),
        "similarity": [stats.PearsonCorr(), stats.Cov()],
    },
}

# Define a dictionary of hyperparameter options for each model, preprocessor, and feature selector
default_config_dict = {
    'models': model_options,
    'preprocessors': preprocessor_options,
    'features': feature_selection_options,
    'hyperparameters': hyperparameters_options
}