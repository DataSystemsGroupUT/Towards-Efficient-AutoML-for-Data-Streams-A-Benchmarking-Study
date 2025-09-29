#!/bin/bash
module load miniconda3/22.11.1

eval "$(conda shell.bash hook)"


conda activate benchmark



while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset_name) dataset_name="$2"; shift ;;
    --seed) seed="$2"; shift ;;
    --model_name) model="$2"; shift ;;  # Added model argument
    --n_model) n_model="$2"; shift ;;  # For HATC, SRPC, ARFC

  esac
  shift
done

# Run baseline experiment
python run_script_2.py --model_name "$model" --dataset_name "$dataset_name" --seed "$seed" --n_model "$n_model" 