import psutil
import time
import tqdm
from river import stream, metrics, linear_model,naive_bayes,tree,neighbors,preprocessing
from ASML import AutoStreamClassifier
from collecter import WindowClassificationPerformanceEvaluator
import pandas as pd
import json
import argparse
import random

import warnings

import os 
import sys

warnings.filterwarnings("ignore")

from codecarbon import OfflineEmissionsTracker

def main(dataset_name, EW=1000, ES=3, B=10,seed=42):
    

    print(
        f"Loading dataset: {dataset_name}, Random Seed:{seed}")
    print(f"Current Hyperparameters: EW - {EW}, ES - {ES}, B - {B}")

    df = pd.read_csv(f"stream_datasets/{dataset_name}.csv")

    x = df.drop("class", axis=1)
    y = df["class"]

    dataset = stream.iter_pandas(x, y)
    file_name = f"AutoStreamML_{dataset_name}_seed_{seed}_budget_{B}_exploration_window_{EW}_ensemble_size_{ES}.json"
    file_name = f"AutoStreamML_{dataset_name}_seed_{seed}_exploration_window_{EW}_ensemble_size_{ES}_budget_{B}.json"
    


    ASC = AutoStreamClassifier(config_dict=None, #config_dict
        exploration_window=EW, # Window Size
        prediction_mode="ensemble", #change 'best' if you want best model prediction 
        budget=B,# How many pipelines run concurrently
        ensemble_size=ES, # Ensemble size 
        metric=metrics.Accuracy(), # Online metrics
        verbose=False,
        seed=seed, # Random/Fixed seed
    )

    online_metric = metrics.Accuracy()
    
    
    # WCPE for plotting the results in line graph
    wcpe = WindowClassificationPerformanceEvaluator(metric=metrics.Accuracy(),
                                                    window_width=1000,
                                                    print_every=1000)

    scores = []
    times = []
    memories = []
    emissions = []
    energy = []
    tracker=OfflineEmissionsTracker(country_iso_code="EST",log_level="critical",allow_multiple_runs=True)#,experiment_id=run_id,save_to_file=True,output_file='emissions.csv')
    tracker.start()




    for x, y in tqdm.tqdm(dataset, leave=True):

        tracker.start_task()
        mem_before = psutil.Process(os.getpid()).memory_info().rss # Recording Memory
        start = time.time()  # Recording Time
        y_pred = ASC.predict_one(x)  # Predict/Test
        s = online_metric.update(y, y_pred).get() # Update Metrics
        # windows Update
        wcpe.update(y, y_pred)
        ASC.learn_one(x, y) # Online Learning
        end = time.time()
        mem_after = psutil.Process(os.getpid()).memory_info().rss

        iteration_mem = mem_after - mem_before
        memories.append(iteration_mem)
        iteration_time = end - start
        emission_record=tracker.stop_task()
        scores.append(s)
        times.append(abs(iteration_time))
        emissions.append(emission_record.emissions)
        energy.append(emission_record.energy_consumed)
    

    



    # saving results in dict
    save_record = {
        "model": "AutoStreamML",
        "dataset": dataset_name,
        "prequential_scores": scores,
        "windows_scores": wcpe.get(),
        "time": times,
        "memory": memories,
        "emission": emissions,
        "energy_consumed": energy #kwh
    }


    # To store the dictionary in a JSON file
    dir_path = "experiment-results/AutoStreamML"
    os.makedirs(dir_path, exist_ok=True) 

    file_name = f"{save_record['model']}_{save_record['dataset']}_seed_{seed}_exploration_window_{EW}_ensemble_size_{ES}_budget_{B}.json"    
    file_path = os.path.join(dir_path, file_name)

    
    # To store the dictionary in a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(save_record, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoStreamML Script")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset file (without extension)")

    parser.add_argument("--exploration_window", type=int, default=1000, help="Exploration Window")
    parser.add_argument("--ensemble_size", type=int, default=3, help="Ensemble Size")
    parser.add_argument("--budget", type=int, default=10, help="Budget")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()
    main(
        args.dataset_name,
        args.exploration_window,
        args.ensemble_size,
        args.budget,
        args.seed
    )
