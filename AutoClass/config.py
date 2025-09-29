from river import tree, neighbors

def range_gen(min_n,max_n,step=1,float_n=False):
    if float_n:
        return [min_n + i * step for i in range(int((max_n - min_n) / step) + 1)]
    return list(range(min_n,max_n+1,step))

model_options = [
    tree.HoeffdingTreeClassifier(),
    tree.HoeffdingAdaptiveTreeClassifier(),
    neighbors.KNNClassifier(),
    neighbors.KNNADWINClassifier(),
]
hyperparameters_options = {
    "HoeffdingTreeClassifier": {
        "grace_period": range_gen(10, 200, step=50),
        "split_confidence": range_gen(0, 1, step=0.01, float_n=True),
    },
    "HoeffdingAdaptiveTreeClassifier": {
        "grace_period": range_gen(10, 200, step=50),
        "split_confidence": range_gen(0, 1, step=0.01, float_n=True),
    },
    "KNNClassifier": {
        "n_neighbors": range_gen(2, 30, step=1),
    },
    "KNNADWINClassifier": {
        "n_neighbors": range_gen(2, 30, step=1),
    },
}

default_config_dict = {}

default_config_dict['algorithms'] = model_options 
default_config_dict['hyperparameters'] = hyperparameters_options