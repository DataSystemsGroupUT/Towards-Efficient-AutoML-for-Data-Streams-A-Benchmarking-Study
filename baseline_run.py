import tqdm
import time
import psutil
import json
import pandas as pd
import sys
from river import metrics,stream,preprocessing,linear_model,ensemble
from river import tree

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import argparse
import random
from collecter import WindowClassificationPerformanceEvaluator

from codecarbon import OfflineEmissionsTracker
import os




# Extract the name of the model from the model class
def extract_model_short_form(model):
    input_string = type(model).__name__
    uppercase_letters = []
    for char in input_string:
        if char.isupper():
            uppercase_letters.append(char)
    return ''.join(uppercase_letters)

def main(dataset_name,model_name,n_model,seed):

    current_time=datetime.now().strftime("%Y-%m-%d_%H-%M")

    run_id=f"{model_name}_nmodel{n_model}_seed{seed}"

    tracker=OfflineEmissionsTracker(country_iso_code="EST",log_level="critical",experiment_id=run_id,save_to_file=True,output_file='emissions.csv',allow_multiple_runs=True)

    # seed = 42 #random.randint(42,52) # Currently, we are using default seed, but you can use a random seed for multiple runs.

    # Selecting a model from a set of baseline models
    if model_name=='HATC':
        model_raw = tree.HoeffdingAdaptiveTreeClassifier(seed=seed)
    # elif model_name=='LBC':
    #     model_raw = ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingAdaptiveTreeClassifier(),seed=seed)
    elif model_name=='SRPC':
        model_raw = ensemble.SRPClassifier(n_models=n_model,seed=seed)
    elif model_name=='ARFC':
        model_raw = ensemble.AdaptiveRandomForestClassifier(n_models=n_model,seed=seed)

    file_name = f"{extract_model_short_form(model_raw)}_{dataset_name}_seed_{seed}_nmodel_{n_model}.json" # file name for save
    file_path = f"experiment-results/{extract_model_short_form(model_raw)}/{file_name}"

# Check if the file exists before proceeding
    # if os.path.exists(file_path):
    #     print(f"File {file_path} already exists. Skipping execution.")
    #     sys.exit(0)


    print(f"Model Name: {extract_model_short_form(model_raw)}")
    
    print(f"Loading dataset: {dataset_name},Random Seed:{seed}")

    # We are using Standerd Scaler for all of the model preprocessing.
    model = preprocessing.StandardScaler() | model_raw

    # Reading Datasets
    df = pd.read_csv(f"stream_datasets/{dataset_name}.csv")
    
    x = df.drop('class', axis=1) 
    y = df['class']
    
    # converting dataframe to stream
    dataset = stream.iter_pandas(x, y)
    
    # storing the results    
    scores = []
    times = []
    memories = []
    metric = metrics.Accuracy()
    emissions = []
    energy = []
    
    # WCPE for plotting the results in line graph
    wcpe = WindowClassificationPerformanceEvaluator(metric=metrics.Accuracy(),
                                                    window_width=1000,
                                                    print_every=1000)
    

    tracker.start()
    # tracker.start_task()
    # for x, y in tqdm.tqdm(dataset,leave=False):
    for x, y in tqdm.tqdm(dataset, leave=False, desc="Processing", dynamic_ncols=True):
        tracker.start_task()
        mem_before = psutil.Process(os.getpid()).memory_info().rss # Recording Memory
        start = time.time() # Recording Time
        try:
            y_pred = model.predict_one(x) # Predict/Test
            s = metric.update(y,y_pred).get() # Update Metrics
            wcpe.update(y, y_pred) # windows Update
            model.learn_one(x, y) # Online Learning
        except:
            s=0
            continue
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
        "model": extract_model_short_form(model_raw),
        "dataset": dataset_name,
        "prequential_scores": scores,
        "windows_scores": wcpe.get(),
        "time": times,
        "memory": memories,
        "emission": emissions,
        "energy_consumed": energy #kwh
    }
    
    print(f"{extract_model_short_form(model_raw)}: Accuracy on {dataset_name}: {metric.get()}")


    dir_path = f"experiment-results/{save_record['model']}"
    os.makedirs(dir_path, exist_ok=True)  # Ensure the directory exists   


    file_name = f"{save_record['model']}_{save_record['dataset']}_seed_{seed}_nmodel_{n_model}.json" 
    file_path = os.path.join(dir_path, file_name)
    
    # Write the dictionary to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(save_record, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Run Script")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset file (without extension)")

    parser.add_argument("--model_name", type=str, help="Name of the dataset file (without extension)")
    parser.add_argument("--n_model", type=int, help="Model ensemble size")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()
    main(args.dataset_name,args.model_name,args.n_model,args.seed)
