# Application script for automated River
# python3 oaml_ensemble.py "forest_cover"  5000 5000 acc acc 60 evol 0

# imports
import os

# imports
import pandas as pd
import sys
import numpy as np
from pathlib import Path

# import gc
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
# import tracemalloc
# tracemalloc.start()
# snapshot1 = tracemalloc.take_snapshot()
# sys.path.append(os.path.abspath('../'))

# from ASML import WindowClassificationPerformanceEvaluator

import warnings
warnings.filterwarnings("ignore")

class WindowClassificationPerformanceEvaluator():
    """Evaluator for tracking classification performance in a window-wise manner.

    This class is designed to evaluate a classification model's performance in a window-wise
    fashion. It uses a specified metric to measure the performance and maintains a list of scores
    calculated at the end of each window.

    Parameters:
    - metric: metrics.base.MultiClassMetric, optional (default=None)
        The metric used to evaluate the model's predictions. If None, the default metric is
        metrics.Accuracy().
    - window_width: int, optional (default=1000)
        The width of the evaluation window, i.e., the number of samples after which the metric is
        calculated and the window is reset.
    - print_every: int, optional (default=1000)
        The interval at which the current metric value is printed to the console.

    Methods:
    - update(y_pred, y, sample_weight=1.0):
        Update the evaluator with the predicted and true labels for a new sample. The metric is
        updated, and if the window is complete, the metric value is added to the scores list.
    - get():
        Get the list of metric scores calculated at the end of each window.

    Example:
    >>> evaluator = WindowClassificationPerformanceEvaluator(
    ...     metric=metrics.Accuracy(),
    ...     window_width=500,
    ...     print_every=500
    ... )
    >>> for x, y in stream:
    ...     y_pred = model.predict(x)
    ...     evaluator.update(y_pred, y)
    ...
    >>> scores = evaluator.get()
    >>> print(scores)

    Note: This class assumes a multi-class classification scenario and is designed to work with
    metrics that inherit from metrics.base.MultiClassMetric.
    """
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
print(f"Run Count",int(sys.argv[8]))
print(f"Ensemble size",int(sys.argv[9]))
print(f"Random seed",int(sys.argv[10]))

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
run_count = int(sys.argv[8])
ensemble_size = int(sys.argv[9])
seed = int(sys.argv[10])
# Data

B = pd.read_csv(data_loc)

# Preprocessing of data: Drop NaNs, check for zero values

file_name = f"OnlineAutoML_{sys.argv[1]}_seed_{seed}_ensemble_size_{ensemble_size}_{time_budget}_initialBatch_{initial_batch}_cache_{sliding_window}.json"

if os.path.exists(f"experiment-results/OAML/{file_name}"):
    print(f"File already exists. Skipping execution.")
    sys.exit(0)

         
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
    post_processing=BestFitOnlinePostProcessing(),
    store="nothing",
    random_state=seed,
    output_directory='gama_output_file',
)


Auto_pipeline.fit(X.iloc[0:initial_batch], y[0:initial_batch])
print(
    f"Initial model is {Auto_pipeline.model} and hyperparameters are: {Auto_pipeline.model._get_params()}",flush=True
)
print("Online model is updated with latest AutoML pipeline.",flush=True)

# Online learning

Backup_ensemble = ensemble.VotingClassifier([Auto_pipeline.model])

Online_model = Backup_ensemble
count_drift = 0
last_training_point = initial_batch

scores_oaml = []
times_oaml= []
memories_oaml = []
emissions = []
energy = []
tracker=OfflineEmissionsTracker(country_iso_code="EST",log_level="critical",allow_multiple_runs=True)#,experiment_id=run_id,save_to_file=True,output_file='emissions.csv')
tracker.start()

wcpe = WindowClassificationPerformanceEvaluator(metric=metrics.Accuracy(),
                                                    window_width=1000,
                                                    print_every=1000)



current_time=datetime.now().strftime("%Y-%m-%d_%H-%M")

run_id=f"OAML_ensemble_size_{ensemble_size}_{time_budget}"

# tracker=OfflineEmissionsTracker(country_iso_code="EST",log_level="critical",experiment_id=run_id,save_to_file=True,output_file='emissions.csv')



for i in tqdm.tqdm(range(initial_batch)):
    tracker.start_task()
    start = time.time()
    y_pred = Online_model.predict_one(X.iloc[i].to_dict())
    online_metric = online_metric.update(y[i], y_pred)
    wcpe.update(y[i], y_pred)
    end = time.time()
    #  mem_after = psutil.Process().memory_info().rss
    #  iteration_mem = mem_after - mem_before
    iteration_time = end - start
    emission_record=tracker.stop_task()
    scores_oaml.append(online_metric.get())
    #  memories_oaml.append(iteration_mem)
    times_oaml.append(iteration_time)
    emissions.append(emission_record.emissions)
    energy.append(emission_record.energy_consumed)
        
# test_counter=0


# skipped=0
print(f"Test batch - 0 with 0",flush=True)
for i in tqdm.tqdm(range(initial_batch + 1, len(B))):
    # test_counter+=1
    # if test_counter>=300:
    #     break
    

    try:

        
#         # Print top 5 memory-consuming lines
#         if i%1000==0:
#             snapshot2 = tracemalloc.take_snapshot()

# # Compare memory usage between snapshots
#             stats = snapshot2.compare_to(snapshot1, 'lineno')
#             for stat in stats[:15]:
#                 print(stat)
        tracker.start_task()
    #   mem_before = psutil.Process().memory_info().rss
        start = time.time()
        # Test then train - by one
        y_pred = Online_model.predict_one(X.iloc[i].to_dict())
   
        online_metric = online_metric.update(y[i], y_pred)
        wcpe.update(y[i], y_pred) #windows Update
        Online_model = Online_model.learn_one(X.iloc[i].to_dict(), int(y[i]))
     
        # Print performance every x interval
     
    #     if i % 1000 == 0:
    #         print(f"Test batch - {i} with {online_metric}",flush=True)
        # Check for drift
     
        drift_detector.add_element(int(y_pred != y[i]))
        if (drift_detector.detected_change()) or ((i - last_training_point) > 50000):
   
            if i - last_training_point < 1000:
                # end = time.time()
                # scores_oaml.append(online_metric.get())
                # iteration_time = end - start
                # times_oaml.append(iteration_time)
   
                continue
          #   if drift_detector.detected_change():
          #       print(
          #           f"Change detected at data point {i} and current performance is at {online_metric}",flush=True
          #       )
          #   if (i - last_training_point) > 50000:
          #       print(
          #           f"No drift but retraining point {i} and current performance is at {online_metric}",flush=True
          #       )
     
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
                post_processing=BestFitOnlinePostProcessing(),
                store="nothing",
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
              #   print("Online model is updated with Backup Ensemble.",flush=True)
            else:
                Online_model = Auto_pipeline.model
              #   print("Online model is updated with latest AutoML pipeline.",flush=True)
     
            # Ensemble update with new model, remove oldest model if ensemble is full
            Backup_ensemble.models.append(Auto_pipeline.model)
            if len(Backup_ensemble.models) > ensemble_size:
                Backup_ensemble.models.pop(0)
            
            print("Ensemble size: ",len(Backup_ensemble) )

            
            # del dataset
            # del Auto_pipeline
            # del Perf_automodel
            # del Perf_ensemble

            # del dataset
            # del Auto_pipeline
            # gc.collect()
     
          #   print(
          #       f"Current model is {Online_model} and hyperparameters are: {Online_model._get_params()}"
          #   ,flush=True)
        end = time.time()
        iteration_time = end - start
        emission_record=tracker.stop_task()
        scores_oaml.append(online_metric.get())
        #  memories_oaml.append(iteration_mem)
        times_oaml.append(iteration_time)
        emissions.append(emission_record.emissions)
        energy.append(emission_record.energy_consumed)

    #     mem_after = psutil.Process().memory_info().rss
    #     iteration_mem = mem_after - mem_before
        iteration_time = end - start
        scores_oaml.append(online_metric.get())
    #     memories_oaml.append(iteration_mem)
        times_oaml.append(iteration_time)
    #   print("Iteration:",i ," finished")
      
    except BrokenPipeError:
      print("Skipped")
    #   skipped+=1
    #   print('BrokenPipeError caught', file=sys.stderr)

      continue
    except Exception as e:
      print("Skipped")
    #   skipped+=1  # This will catch any other exceptions
    #   print(f'An unexpected error occurred: {e}')
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
    
    
#file_name = f"{save_record['model']}_{save_record['dataset']}.json"
file_name = f"{save_record['model']}_{save_record['dataset']}_seed_{seed}_ensemble_size_{ensemble_size}_{time_budget}_initialBatch_{initial_batch}_cache_{sliding_window}.json"
# print('Skpped: ',skipped)
#json_folder_path = r"saved_results_json"
json_folder_path = "/gpfs/helios/home/etais/hpc_alimahar/benchmark/ASML-CLS/experiment-results/OAML" # change temp to  saved_results_json for final run

# To store the dictionary in a JSON file
with open(f"{json_folder_path}/{file_name}", 'w') as json_file:
    json.dump(save_record, json_file)

print("Result Saved :", file_name)
sys.exit(0)
# exit()