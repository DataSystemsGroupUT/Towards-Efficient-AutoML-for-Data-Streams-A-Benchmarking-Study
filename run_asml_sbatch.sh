#!/bin/bash
module load miniconda3/22.11.1

eval "$(conda shell.bash hook)"


conda activate benchmark



while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset_name) dataset_name="$2"; shift ;;
    --seed) seed="$2"; shift ;;
    --ASML_exploration_window) exploration_window="$2"; shift ;;
    --ASML_ensemble_size) ensemble_size="$2"; shift ;;
    --ASML_budget) budget="$2"; shift ;;
  esac
  shift
done

# Run the ASML experiment
python run_script_2.py --model_name asml --dataset_name "$dataset_name" --ASML_exploration_window "$exploration_window" --ASML_ensemble_size "$ensemble_size" --ASML_budget "$budget" --seed "$seed" 