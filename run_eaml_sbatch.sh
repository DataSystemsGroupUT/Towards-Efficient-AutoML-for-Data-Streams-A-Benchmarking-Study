#!/bin/bash
module load miniconda3/22.11.1

eval "$(conda shell.bash hook)"


conda activate benchmark



# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset_name) dataset_name="$2"; shift ;;
    --seed) seed="$2"; shift ;;
    --EAML_population_size) population_size="$2"; shift ;;
    --EAML_sampling_size) sampling_size="$2"; shift ;;
    --EAML_sampling_rate) sampling_rate="$2"; shift ;;
  esac
  shift
done

# Run the EAML experiment
python run_script_2.py --model_name eaml --dataset_name "$dataset_name" --EAML_population_size "$population_size" --EAML_sampling_size "$sampling_size" --EAML_sampling_rate "$sampling_rate" --seed "$seed" 
