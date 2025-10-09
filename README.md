# Towards Efficient AutoML for Data Streams: A Benchmarking Study 
Benchmarking AutoML solutions for streaming classification with a standardized online evaluation methodology.


Repository hosted at below Github repo <br>
https://github.com/DataSystemsGroupUT/Towards-Sustainable-and-Scalable-AutoML-for-Data-Streams-A-Multi-Metric-Benchmarking-Study   
<br>
In order download repository either by using GUI of Github by click 'Code' button then 'Download zip' button or by using below cli command 
```
git clone https://github.com/DataSystemsGroupUT/Towards-Sustainable-and-Scalable-AutoML-for-Data-Streams-A-Multi-Metric-Benchmarking-Study 
```

# Requirements:
- Operating System: Linux (experiments run on RHEL9, analysis of results on Debian 12)  
- Conda (Miniconda or Anaconda)

### Recommended:
- HPC with SLURM (for large-scale automated experiments)  

# Setting python environments
### Prerequisite for HPC:
In HPC in order setup the python environments there is need to log in HPC over SSH and run srun command
```
srun --partition=main -A "[YOUR-ACCOUNT-NAME]" --cpus-per-task=2 --mem=2G --time=2:00:00 --pty bash
```
After that step setting python environment contain same step for both SLURM and non-SLURM environments.
Use the given yml file to create a conda environment for running experiments:
```
conda env create -f benchmark.yml 
conda env create -f oaml_benchmark.yml 
```
Once environments created it canbe activated with:
```
conda activate benchmark
conda activate oaml_benchmark
```
As optional step, below command can be executed to be sure all necessary scripts have permission to run.
```
find . -type f \( -name "*.sh" -o -name "*.py" \) -exec chmod +x {} \;
```
Environment 'benchmark' can run experiments on all framework except OnlineAutoML. For OnlineAutoML 'oaml_benchmark' should be used.
# Running experiments
## On SLURM environment(recommended)
Before running experiments take look at 'sbatch_config.env' file:
```
export SBATCH_PARTITION="[YOUR-PARTITION-NAME]"
export SBATCH_ACCOUNT="[YOUR-ACCOUNT-NAME]"
export SBATCH_CPUS_PER_TASK=12
export SBATCH_MEM="32G"
export SBATCH_TIME="2-14:30:00"
```
Replace [YOUR-ACCOUNT-NAME] and [YOUR-PARTITION-NAME] and with your actual HPC account and partitio names. Rest of values used to adjust hardware resources and maximum time limit for experiments.
To run experiments automatically. 
```
./run_experiments
```
This command will submit given experiments in isolated containers.
## Non-SLURM environment
For NON-SLURM environments there is no large scale automation designed. But still experiments can be run with 'run_script_2.py'
In order to do this first activate python environments. 
```
conda activate benchmark
```
After activating environment run 'run_script_2.py'. Example run:
```
python run_script_2.py --model_name asml --dataset_name adult --ASML_exploration_window 1001 --ASML_ensemble_size 3 --ASML_budget 10 --seed 42 
```
Instructions has been given in below section.
### Possible options for single run:
```
General Arguments
    --model_name (str): Algorithm to run

        Choices: asml, ac, oaml, eaml, hatc, srpc, arfc

    --dataset_name (str): Dataset to use from 'stream_datasets' directory.

        Example values: electricity, adult, etc.

    --seed (int): Random seed for reproducibility

ARFC and SRPC Specific Arguments
    --n_model (int): Size of the model ensemble

OAML-Specific Arguments

    --OAML_initial (int): Initial sample size

    --OAML_cache (int): Initial cache size

    --OAML_time_budget (int): Time budget

    --OAML_ensemble_size (int): Ensemble size

AC-Specific Arguments

    --AC_exploration_window (int): Exploration window size

    --AC_population_size (int): Population size

ASML-Specific Arguments

    --ASML_exploration_window (int): Exploration window size

    --ASML_ensemble_size (int): Ensemble size

    --ASML_budget (int): Budget

EAML-Specific Arguments

    --EAML_population_size (int): Population size

    --EAML_sampling_size (int): Sampling size

    --EAML_sampling_rate (int): Sampling rate
```

## Analysis of results:
The visualization and statistical tests were performed using Jupyter notebooks.
The following YAML file contains the necessary libraries for visualization and statistical analysis:
```
conda env create -f orange33.yml
```
To run Jupyter server with created python environment:
```
conda activate orange33
jupyter notebook --notebook-dir=./
```
### Data generation
The study includes a synthetic dataset generation process. The generated datasets are stored in the 'stream_datasets/' directory, alongside the other datasets. <br>
Additionally, the code used to generate the synthetic data can be found in the 'data_gen.ipynb' notebook. The notebook does not require a specific Python environment, as the commands to install the necessary packages are provided within the notebook.


### Raw results.
Due to the size of the raw results (~75GB), we could not include them in the GitHub repository. However, they can be downloaded as .zip file: <br>
https://f003.backblazeb2.com/file/7rTSHacdvd/experiment-results.zip  <br>
After download extract zip file and replace current 'experiment_results' directory with directory from .zip file. <br>
Zip file is ~25GB and the extracted contents are ~75GB. Therefore, more than 100 GB of free space is needed to reliably download and extract the file.
