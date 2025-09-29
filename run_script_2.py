import argparse
import concurrent.futures
import subprocess



# HATC: HoeffdingA daptive Tree Classifier
# SRPC: Streaming Random Patches ensemble classifier
# ARFC: Adaptive Random Forest classifier
# OAML: Online AutoML
# AC:   AutoClass
# ASML: AutoStreamML
# EAML: EvoAutoML


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run different scripts on datasets.")
parser.add_argument('--model_name', type=str, help='Short name of the script to run (asml, ac, oaml, eaml, hatc, srpc, and arfc)')
parser.add_argument('--dataset_name', type=str, help='Name of the dataset to run the script on (e.g., electricity, adult)')


#baseline
parser.add_argument("--n_model", type=int, help="Model ensemble size")
parser.add_argument("--seed", type=int, help="Random seed")

#OAML
parser.add_argument("--OAML_initial", type=int, help="OAML Initial sample size")
parser.add_argument("--OAML_cache", type=int, help="OAML Initial cache size")
parser.add_argument("--OAML_time_budget", type=int, help="OAML time budget")
parser.add_argument("--OAML_ensemble_size", type=int, help="OAML ensemble size")

#AC
parser.add_argument("--AC_exploration_window", type=int, help="AC window size")
parser.add_argument("--AC_population_size", type=int, help="AC model")

#ASML
parser.add_argument("--ASML_exploration_window", type=int, help="ASML window size")
parser.add_argument("--ASML_ensemble_size", type=int, help="ASML ensemble size")
parser.add_argument("--ASML_budget", type=int, help="ASML budget")

#EAML
parser.add_argument("--EAML_population_size", type=int, help="EAML population size")
parser.add_argument("--EAML_sampling_size", type=int, help="EAML sampling size")
parser.add_argument("--EAML_sampling_rate", type=int, help="EAML sampling rate")

args = parser.parse_args()

model_name=args.model_name
dataset_name=args.dataset_name

n_model=args.n_model
seed=args.seed

AC_exploration_window=args.AC_exploration_window
AC_population_size=args.AC_population_size


if model_name in ['hatc', 'srpc', 'arfc']:
    command = [
        'python', 'baseline_run.py', dataset_name,  '--model_name', model_name.upper(), '--n_model', str(n_model), '--seed', str(seed)
    ]
elif model_name == 'oaml':
    command = [
        'python', 'OAML/oaml_ensemble.py', dataset_name, str(args.OAML_initial), str(args.OAML_cache), 'acc', 'acc', str(args.OAML_time_budget), 'evol', str(args.OAML_ensemble_size), str(args.seed)
    ]
elif model_name == 'ac':
    command = [
        'python', 'ac_run.py', dataset_name,  '--exploration_window', str(AC_exploration_window) ,'--population_size', str(AC_population_size), '--seed', str(args.seed)
        ]
elif model_name == 'asml':
    command = [
        'python', 'asml_run.py', dataset_name,  '--exploration_window', str(args.ASML_exploration_window), '--ensemble_size', str(args.ASML_ensemble_size), '--budget', str(args.ASML_budget), '--seed', str(args.seed)
        ]
elif model_name == 'eaml':
    command = [
        'python', 'eaml_run.py', dataset_name, 
        '--population_size', str(args.EAML_population_size),
        '--sampling_size', str(args.EAML_sampling_size),
        '--sampling_rate', str(args.EAML_sampling_rate),
        '--seed', str(args.seed)
    ]
else:
    raise ValueError("Invalid script name provided.")
try:
    # Redirect output to the terminal directly
    process = subprocess.run(command, check=True)
    print(f"Dataset {dataset_name}, Script {model_name} : Completed Successfully")
except subprocess.CalledProcessError as e:
    print(f"Dataset {dataset_name}, Script {model_name} : Error\n{e}")





