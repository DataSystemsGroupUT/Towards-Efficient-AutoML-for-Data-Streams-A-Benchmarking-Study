#!/bin/bash
module load miniconda3/22.11.1

eval "$(conda shell.bash hook)"


conda activate benchmark


while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset_name) dataset_name="$2"; shift ;;
    --seed) seed="$2"; shift ;;
    --AC_exploration_window) exploration_window="$2"; shift ;;
    --AC_population_size) population_size="$2"; shift ;;
  esac
  shift
done

# Run the AC experiment
python run_script_2.py --model_name ac --dataset_name "$dataset_name" --AC_exploration_window "$exploration_window" --AC_population_size "$population_size" --seed "$seed" 


  
