#!/bin/bash
module load miniconda3/22.11.1

eval "$(conda shell.bash hook)"


conda activate oaml_benchmark


# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset_name) dataset_name="$2"; shift ;;
    --seed) seed="$2"; shift ;;
    --OAML_initial) OAML_initial="$2"; shift ;;
    --OAML_cache) OAML_cache="$2"; shift ;;
    --OAML_time_budget) OAML_time_budget="$2"; shift ;;
    --OAML_ensemble_size) OAML_ensemble_size="$2"; shift ;;
  esac
  shift
done

# Run OAML experiment
python run_script_2.py --model_name oaml --dataset_name "$dataset_name" --seed "$seed" \
  --OAML_initial "$OAML_initial" --OAML_cache "$OAML_cache" \
  --OAML_time_budget "$OAML_time_budget" --OAML_ensemble_size "$OAML_ensemble_size" 
