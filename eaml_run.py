import tqdm
import time
import psutil
import pandas as pd
import random

from river import metrics,stream
from EvOAutoML import classification
from collecter import WindowClassificationPerformanceEvaluator
from codecarbon import OfflineEmissionsTracker


import json

import argparse

import warnings
import os 
import sys

warnings.filterwarnings("ignore")

def main(dataset_name, population_size, sampling_size, sampling_rate, seed):



    file_name = f"EvoAutoML_{dataset_name}_seed_{seed}_population_size_{population_size}_sampling_size_{sampling_size}_sampling_rate_{sampling_rate}.json"

    # if os.path.exists(f"experiment-results/EvoAutoML/{file_name}"):
    #      print(f"File already exists. Skipping execution.")
    #      sys.exit(0)
 
    print(f"Loading dataset: {dataset_name}, Random Seed:{seed}")


    df = pd.read_csv(f"stream_datasets/{dataset_name}.csv")

    x = df.drop('class', axis=1) 
    y = df['class']

    dataset = stream.iter_pandas(x, y)
    
    model = classification.EvolutionaryBaggingClassifier(
        population_size=population_size,
        sampling_size=sampling_size,
        sampling_rate=sampling_rate,
        metric=metrics.Accuracy,
        seed=seed
    )

    metric = metrics.Accuracy()
    
    wcpe = WindowClassificationPerformanceEvaluator(metric=metrics.Accuracy(),
                                                    window_width=1000,
                                                    print_every=1000)

    scores_evo = []
    times_evo = []
    memories_evo = []
    emissions = []
    energy = []
    tracker=OfflineEmissionsTracker(country_iso_code="EST",log_level="critical",allow_multiple_runs=True)#,experiment_id=run_id,save_to_file=True,output_file='emissions.csv')
    tracker.start()





    for x, y in tqdm.tqdm(dataset, leave=True):

        tracker.start_task()
        mem_before = psutil.Process(os.getpid()).memory_info().rss
        start = time.time()

        y_pred = model.predict_one(x)  # make a prediction
        metric.update(y, y_pred)  # update the metric
        wcpe.update(y, y_pred) #windows Update
        model = model.learn_one(x,y)  # make the model learn
        
        end = time.time()
        mem_after = psutil.Process(os.getpid()).memory_info().rss

        iteration_mem = mem_after - mem_before
        memories_evo.append(iteration_mem)

        iteration_time = end - start
        emission_record=tracker.stop_task()

        scores_evo.append(metric.get())
        times_evo.append(abs(iteration_time))
        emissions.append(emission_record.emissions)
        energy.append(emission_record.energy_consumed)


    

    
    save_record = {
        "model": "EvoAutoML",
        "dataset": dataset_name,
        "prequential_scores": scores_evo,
        "windows_scores": wcpe.get(),
        "time": times_evo,
        "memory": memories_evo,
        "emission": emissions,
        "energy_consumed": energy #kwh
    }
    



    #file_name = f"{save_record['model']}_{save_record['dataset']}.json"
    # file_name = f"{save_record['model']}_{save_record['dataset']}_population_size_{population_size}_sampling_size_{sampling_size}_sampling_rate_{sampling_rate}.json"
    
    #print("Result Saved path:",file_name)
    
    # To store the dictionary in a JSON file

    dir_path = "experiment-results/EvoAutoML"
    os.makedirs(dir_path, exist_ok=True)  # Make sure the directory exists


    file_path = os.path.join(dir_path, file_name)
    
    # To store the dictionary in a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(save_record, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoAutoMl Script")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset file (without extension)")
    parser.add_argument("--population_size", type=int, default=10, help="Population size for the model (default: 10)")
    parser.add_argument("--sampling_size", type=int, default=1, help="Sampling size for the model (default: 1)")
    parser.add_argument("--sampling_rate", type=int, default=1000, help="Sampling rate for the model (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,

        population_size=args.population_size,
        sampling_size=args.sampling_size,
        sampling_rate=args.sampling_rate,
        seed=args.seed
    )
