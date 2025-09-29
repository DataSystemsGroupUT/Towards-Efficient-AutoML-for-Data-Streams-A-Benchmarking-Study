import psutil
import time
import tqdm
from river import stream, metrics
from AutoClass import AutoClass
from collecter import WindowClassificationPerformanceEvaluator
import pandas as pd
import json
import argparse
import random
import os

import warnings

from codecarbon import OfflineEmissionsTracker
import sys
warnings.filterwarnings("ignore")


def main(dataset_name, EW=1000, PS=10,seed=42):
    # seed = 42 #random.randint(42,52) # Currently, we are using default seed, but you can use a random seed for multiple runs.

    print(
        f"Loading dataset: {dataset_name}, Random Seed:{seed}")
    print(f"Current Hyperparameters: EW - {EW}, PS - {PS}")

    # Reading Datasets
    df = pd.read_csv(f"stream_datasets/{dataset_name}.csv")

    x = df.drop("class", axis=1)
    y = df["class"]

    # converting DataFrame to stream
    dataset = stream.iter_pandas(x, y)

    AC = AutoClass(config_dict=None, #config_dict
                    exploration_window=EW, #window size
                    population_size=PS, # how many model run concurrently
                    metric=metrics.Accuracy(), # performance metrics
                    seed=seed) # random seed

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
        mem_before = psutil.Process(os.getpid()).memory_info().rss  # Recording Memory
        tracker.start_task()
        start = time.time() 
        y_pred = AC.predict_one(x) 
        s = online_metric.update(y, y_pred).get() 
        # windows Update
        wcpe.update(y, y_pred)
        AC.learn_one(x, y)  # Online Learning
        end = time.time()


        iteration_time = end - start
        mem_after = psutil.Process(os.getpid()).memory_info().rss
        iteration_mem = mem_after - mem_before
        memories.append(iteration_mem)
        emission_record=tracker.stop_task()
        scores.append(s)
        times.append(abs(iteration_time))
        emissions.append(emission_record.emissions)
        energy.append(emission_record.energy_consumed)
    



    
    # saving results in dict
    save_record = {
        "model": "AutoClass",
        "dataset": dataset_name,
        "prequential_scores": scores,
        "windows_scores": wcpe.get(),
        "time": times,
        "memory": memories,
        "emission": emissions,
        "energy_consumed": energy #kwh
    }

    dir_path = "experiment-results/AutoClass"
    os.makedirs(dir_path, exist_ok=True)  # Make sure the directory exists
    


    file_name = f"{save_record['model']}_{save_record['dataset']}_seed_{seed}_population_size_{PS}_exploration_window_{EW}.json"
    file_path = os.path.join(dir_path, file_name)
    
    # To store the dictionary in a JSON file
    with open(file_path, 'w') as json_file:
    #with open(f"{file_name}", 'w') as json_file:
        json.dump(save_record, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoStreamML Script")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset file (without extension)")

    parser.add_argument("--exploration_window", type=int, help="Exploration Window") #d1000
    parser.add_argument("--population_size", type=int, help="Population Size") #d 10
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()
    main(
        args.dataset_name,
        args.exploration_window,
        args.population_size,
        args.seed
    )
