
import os

# imports
import pandas as pd
import sys
import numpy as np
from pathlib import Path

import gc
from gama import GamaClassifier
from gama.search_methods import AsyncEA
from gama.search_methods import RandomSearch
from gama.search_methods import AsynchronousSuccessiveHalving
from gama.postprocessing import BestFitOnlinePostProcessing

from river import metrics
from river import evaluate
from river import stream
from river import ensemble

from skmultiflow import drift_detection

import time
# import psutil
import json
import tqdm

from codecarbon import OfflineEmissionsTracker

from datetime import datetime

import sys


import warnings
warnings.filterwarnings("ignore")

def get_memory_usage():
    """Returns a DataFrame of all variables in memory along with their sizes."""
    all_vars = {name: sys.getsizeof(obj) for name, obj in globals().items()}
    df = pd.DataFrame(all_vars.items(), columns=["Variable", "Size (bytes)"])
    df = df.sort_values(by="Size (bytes)", ascending=False)
    return df

import gc
import sys
import pandas as pd


def get_object_memory_usage():
    """Returns a DataFrame of all objects in memory with their sizes."""
    objects = gc.get_objects()
    object_sizes = {}

    for obj in objects:
        try:
            obj_type = type(obj).__name__
            obj_size = sys.getsizeof(obj)
            if obj_type in object_sizes:
                object_sizes[obj_type] += obj_size
            else:
                object_sizes[obj_type] = obj_size
        except:
            continue  # Skip objects that cause errors

    df = pd.DataFrame(object_sizes.items(), columns=["Object Type", "Total Size (bytes)"])
    df = df.sort_values(by="Total Size (bytes)", ascending=False)
    return df

import psutil
import os

def kill_child_processes():
    current_pid = os.getpid()  # Get current process ID
    current_process = psutil.Process(current_pid)  # Get process object
    
    children = current_process.children(recursive=True)  # List all child processes
    if not children:
        # print("No child processes found.")
        return

    # print(f"Found {len(children)} child processes. Killing them...")
    
    for child in children:
        # print(f"Killing PID {child.pid}, Status: {child.status()}")
        try:
            child.terminate()  # Gracefully terminate
            child.wait(timeout=5)  # Wait for it to exit
        except psutil.NoSuchProcess:
            pass  # Process already exited
        except psutil.TimeoutExpired:
            # print(f"PID {child.pid} did not exit, force killing...")
            child.kill()  # Force kill if still running

    # print("All child processes terminated.")






class WindowClassificationPerformanceEvaluator():

    def __init__(self, metric=None, window_width=1000, print_every=1000):
        self.window_width = window_width
        self.metric = metric if metric is not None else metrics.Accuracy()
        self.print_every = print_every
        self.counter = 0
        self.scores_list = []
    
    def __repr__(self):
        """Return the class name along with the current value of the metric."""
        metric_value = np.mean(self.get()) * 100
        return f"{self.__class__.__name__}({self.metric.__class__.__name__}): {metric_value:.2f}%"


    def update(self, y_pred, y, sample_weight=1.0):
        """Update the evaluator with new predictions and true labels.

        Parameters:
        - y_pred: Predicted label for the current sample.
        - y: True label for the current sample.
        - sample_weight: Weight assigned to the current sample (default=1.0).
        """
        self.metric.update(y_pred, y, sample_weight=sample_weight)
        self.counter += 1

        # if self.counter % self.print_every == 0:
            # print(f"[{self.counter}] - {self.metric}")

        if self.counter % self.window_width == 0:
            self.scores_list.append(self.metric.get())
            self.metric = type(self.metric)()  # resetting using the same metric type

    def get(self):
        """Get the list of metric scores calculated at the end of each window."""
        return self.scores_list

# Metrics
gama_metrics = {
    "acc": "accuracy",
    "b_acc": "balanced_accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "rmse": "rmse",
}

online_metrics = {
    "acc": metrics.Accuracy(),
    "b_acc": metrics.BalancedAccuracy(),
    "f1": metrics.F1(),
    "roc_auc": metrics.ROCAUC(),
    "rmse": metrics.RMSE(),
}

# Search algorithms
search_algs = {
    "random": RandomSearch(),
    "evol": AsyncEA(),
    "s_halving": AsynchronousSuccessiveHalving(),
}

# User parameters
print(sys.argv[0])  # prints python_script.py
print(f"Data stream is {sys.argv[1]}.")  # prints dataset no
print(f"Initial batch size is {int(sys.argv[2])}.")  # prints initial batch size
print(f"Sliding window size is {int(sys.argv[3])}.")  # prints sliding window size
print(
    f"Gama performance metric is {gama_metrics[str(sys.argv[4])]}."
)  # prints gama performance metric
print(
    f"Online performance metric is {online_metrics[str(sys.argv[5])]}."
)  # prints online performance metric
print(f"Time budget for GAMA is {int(sys.argv[6])}.")  # prints time budget for GAMA
print(
    f"Search algorithm for GAMA is {search_algs[str(sys.argv[7])]}."
)  # prints search algorithm for GAMA

print(f"Ensemble size",int(sys.argv[8]))
print(f"Random seed",int(sys.argv[9]))

current_path = Path.cwd() / "stream_datasets"
data_loc = current_path / f"{sys.argv[1]}.csv"

#data_loc = f"{folder_path}/{sys.argv[1]}.csv"
initial_batch = int(sys.argv[2])  # initial set of samples to train automl
sliding_window = int(
    sys.argv[3]
)  # update set of samples to train automl at drift points (must be smaller than or equal to initial batch size
gama_metric = gama_metrics[
    str(sys.argv[4])
]  # gama metric to evaluate in pipeline search
online_metric = online_metrics[
    str(sys.argv[5])
]  # river metric to evaluate online learning
time_budget = int(sys.argv[6])  # time budget for gama run
search_alg = search_algs[str(sys.argv[7])]
drift_detector = drift_detection.EDDM()  # multiflow drift detector
# drift_detector = EDDM()                            #river drift detector - issues

ensemble_size = int(sys.argv[8])
seed = int(sys.argv[9])
# Data

B = pd.read_csv(data_loc)

# Preprocessing of data: Drop NaNs, check for zero values

file_name = f"OnlineAutoML_{sys.argv[1]}_seed_{seed}_ensemble_size_{ensemble_size}_{time_budget}_initialBatch_{initial_batch}_cache_{sliding_window}.json"

# if os.path.exists(f"experiment-results/OAML/{file_name}"):
#     print(f"File already exists. Skipping execution.")
#     sys.exit(0)

         
if pd.isnull(B.iloc[:, :]).any().any():
    print(
        "Data X contains NaN values. The rows that contain NaN values will be dropped."
    )
    #B.dropna(inplace=True)

if B[:].iloc[:, 0:-1].eq(0).any().any():
    print(
        "Data contains zero values. They are not removed but might cause issues with some River learners."
    )

X = B.copy()
y = X.pop("class")

# Algorithm selection and hyperparameter tuning

Auto_pipeline = GamaClassifier(
    max_total_time=time_budget,
    scoring=gama_metric,
    search=search_alg,
    online_learning=True,
    n_jobs=12,
    post_processing=BestFitOnlinePostProcessing(),
    store="models",
    random_state=seed,
    output_directory='gama_output_file',
)


Auto_pipeline.fit(X.iloc[0:initial_batch], y[0:initial_batch])



print(
    f"Initial model is {Auto_pipeline.model} and hyperparameters are: {Auto_pipeline.model._get_params()}",flush=True
)
print("Online model is updated with latest AutoML pipeline.",flush=True)



Backup_ensemble = ensemble.VotingClassifier([Auto_pipeline.model])

Online_model = Backup_ensemble

last_training_point = initial_batch

scores_oaml = []
times_oaml= []
memories_oaml = []
emissions = []
energy = []
tracker=OfflineEmissionsTracker(country_iso_code="EST",log_level="critical",allow_multiple_runs=True)
tracker.start()

wcpe = WindowClassificationPerformanceEvaluator(metric=metrics.Accuracy(),
                                                    window_width=1000,
                                                    print_every=1000)


current_time=datetime.now().strftime("%Y-%m-%d_%H-%M")

run_id=f"OAML_ensemble_size_{ensemble_size}_{time_budget}"




for i in tqdm.tqdm(range(initial_batch)):
    tracker.start_task()
    mem_before = psutil.Process(os.getpid()).memory_info().rss
    start = time.time()
    y_pred = Online_model.predict_one(X.iloc[i].to_dict())
    online_metric = online_metric.update(y[i], y_pred)
    wcpe.update(y[i], y_pred)
    end = time.time()
    mem_after = psutil.Process(os.getpid()).memory_info().rss
    iteration_mem = mem_after - mem_before

    iteration_time = end - start
    emission_record=tracker.stop_task()
    scores_oaml.append(online_metric.get())

    times_oaml.append(iteration_time)
    emissions.append(emission_record.emissions)
    energy.append(emission_record.energy_consumed)
    memories_oaml.append(iteration_mem)
        





for i in tqdm.tqdm(range(initial_batch + 1, len(B))):

 
    try:
        tracker.start_task()
        mem_before = psutil.Process(os.getpid()).memory_info().rss
        start = time.time()

        y_pred = Online_model.predict_one(X.iloc[i].to_dict())

        online_metric = online_metric.update(y[i], y_pred)
        wcpe.update(y[i], y_pred) 
        Online_model=Online_model.learn_one(X.iloc[i].to_dict(), int(y[i]))
     
         
        drift_detector.add_element(int(y_pred != y[i]))
        if (drift_detector.detected_change()) or ((i - last_training_point) > 5000):
           
            if i - last_training_point < 1000:
                continue

     
            last_training_point = i

     
            # Sliding window at the time of drift
            X_sliding = X.iloc[(i - sliding_window) : i].reset_index(drop=True)
            y_sliding = y[(i - sliding_window) : i].reset_index(drop=True)
     
            # re-optimize pipelines with sliding window
            Auto_pipeline = GamaClassifier(
                max_total_time=time_budget,
                scoring=gama_metric,
                search=search_alg,
                online_learning=True,
                n_jobs=12,
                post_processing=BestFitOnlinePostProcessing(),
                store="models",
                random_state=seed,
                output_directory='gama_output_file',
            )

            Auto_pipeline.fit(X_sliding, y_sliding)
     
            # Ensemble performance comparison
            dataset = []
            for xi, yi in stream.iter_pandas(X_sliding, y_sliding):
                dataset.append((xi, yi))
     
            Perf_ensemble = evaluate.progressive_val_score(
                dataset, Backup_ensemble, metrics.Accuracy()
            )
            Perf_automodel = evaluate.progressive_val_score(
                dataset, Auto_pipeline.model, metrics.Accuracy()
            )
   
            if Perf_ensemble.get() > Perf_automodel.get():
                Online_model = Backup_ensemble

            else:
                Online_model = Auto_pipeline.model

     
            # Ensemble update with new model, remove oldest model if ensemble is full
            Backup_ensemble.models.append(Auto_pipeline.model)
            if len(Backup_ensemble.models) > ensemble_size:
                Backup_ensemble.models.pop(0)


            dataset=None
            X_sliding=None
            y_sliding=None
            Perf_automodel=None
            Perf_ensemble=None

            del dataset
            del X_sliding
            del y_sliding
            del Perf_automodel
            del Perf_ensemble

            kill_child_processes()


        end = time.time()
        iteration_time = end - start
        mem_after = psutil.Process(os.getpid()).memory_info().rss
        iteration_mem = mem_after - mem_before
        emission_record=tracker.stop_task()

        scores_oaml.append(online_metric.get())
        times_oaml.append(iteration_time)
        emissions.append(emission_record.emissions)
        energy.append(emission_record.energy_consumed)
        memories_oaml.append(iteration_mem)


      
    except BrokenPipeError:

        dataset=None
        X_sliding=None
        y_sliding=None
        Perf_automodel=None
        Perf_ensemble=None

        del dataset
        del X_sliding
        del y_sliding
        del Perf_automodel
        del Perf_ensemble

        kill_child_processes()
        


        continue
    except Exception as e:

        dataset=None
        X_sliding=None
        y_sliding=None
        Perf_automodel=None
        Perf_ensemble=None

        del dataset
        del X_sliding
        del y_sliding
        del Perf_automodel
        del Perf_ensemble
        


        kill_child_processes()
   

        continue



save_record = {
        "model": "OnlineAutoML",
        "dataset": sys.argv[1],
        "prequential_scores": scores_oaml,
        "windows_scores": wcpe.get(),
        "time": times_oaml,
        "memory": memories_oaml,
        "emission": emissions,
        "energy_consumed": energy #kwh
    }
    
    

file_name = f"{save_record['model']}_{save_record['dataset']}_seed_{seed}_ensemble_size_{ensemble_size}_time_budget_{time_budget}_initialBatch_{initial_batch}_cache_{sliding_window}.json"


json_folder_path = f"{Path.cwd()}/experiment-results/OnlineAutoML" 
# os.mkdir(parents=True, exist_ok=True) 
os.makedirs(json_folder_path, exist_ok=True)
# To store the dictionary in a JSON file
with open(f"{json_folder_path}/{file_name}", 'w') as json_file:
    json.dump(save_record, json_file)

print("Result Saved :", file_name)
sys.exit(0)

