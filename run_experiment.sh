#!/bin/bash
set -a
source ./sbatch_config.env
set +a

echo "Partition: $SBATCH_PARTITION"
echo "CPU: $SBATCH_CPUS_PER_TASK"
echo "Memory: $SBATCH_MEM"
echo "Time limit: $SBATCH_TIME"


datasets=(
	  adult
	  electricity
	  forest_cover 
	  hyperplane_high_gradual_drift 
	  insects
	  movingRBF
	  moving_squares
	  new_airlines 
	  sea_high_abrupt_drift 
	  sea_high_mixed_drift 
	  shuttle 
	  synth_RandomRBFDrift 
	  synth_agrawal 
	  vehicle_sensIT  old
    rialto
    river_aggrawal
    weather
    river_Waveform_2
    river_anomalysine
    river_Waveform
    skmul_aggrawal
    skmul_anomalysine_2
    river_hyperlane_2
    skmul_anomalysine
    river_hyperlane
    skmul_hyperlane
    river_led   
    skmul_ledgenerator
    river_led_default_drift
    skmul_ledgenerator_default_drift
    river_mixed
    skmul_randomRBF_default_drift
    river_SEA_2
    skmul_randomRBF_gradual
    skmul_randomTree_2
    river_randomRBF_2 
    skmul_randomTree
    river_randomRBF
    river_randomRBF_drift
    skmul_SEA_2
    river_randomTree_2
    skmul_SEA
    river_randomTree
    skmul_Sine
    river_SEA
    skmul_STAGGER_2
    river_SINE
    skmul_STAGGER
    river_STAGGER
    skmul_Waveform
)

models=(ac eaml oaml asml hatc arfc srpc)



seeds=(42 58 72 50 63 123 456 52 31 69 87 14 92 37 105 78 243 88 19 333 76 512 97 401 289 222 654 32 71 999 210 485 777 25 36 159 9999 44 120 808)


n_model_options=(3 2 4 5)


oaml_cache_options=(1000 2000 3000 5000)
oaml_time_budget_options=(30 60 90)
oaml_ensemble_size_options=(5 10 15)

ac_exploration_window_options=(1000 2000 3000)
ac_population_size_options=(5 10 20)

asml_exploration_window_options=(1000 2000 3000)
asml_ensemble_size_options=(1 2 3 4)
asml_budget_options=(5 10 15)

asml_exploration_window_options=(1000 2000 3000)
asml_ensemble_size_options=(2 3 4 5)
asml_budget_options=(5 10 15)

eaml_population_size_options=(5 10 20)
eaml_sampling_size_options=(1 2 3)
eaml_sampling_rate_options=(1000 2000 3000)




echo "Experiment start"
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do

      # HATC, SRPC, ARFC: Iterate over n_model
      if [[ "$model" == "hatc" || "$model" == "srpc" || "$model" == "arfc" ]]; then
        for n_model in "${n_model_options[@]}"; do
          echo "Submitting Baseline: model=$model dataset=$dataset seed=$seed n_model=$n_model"
          sbatch --partition="$SBATCH_PARTITION" --cpus-per-task="$SBATCH_CPUS_PER_TASK" --mem="$SBATCH_MEM" --time="$SBATCH_TIME" \
          run_baseline_sbatch.sh --dataset_name "$dataset" --seed "$seed" --model_name "$model" --n_model "$n_model"
        done
      fi

      if [[ "$model" == "oaml" ]]; then
          for cache in "${oaml_cache_options[@]}"; do
            for time_budget in "${oaml_time_budget_options[@]}"; do
              for ensemble_size in "${oaml_ensemble_size_options[@]}"; do
                echo "Submitting OAML: dataset=$dataset seed=$seed initial=$initial cache=$cache time_budget=$time_budget ensemble_size=$ensemble_size"
                sbatch --partition="$SBATCH_PARTITION" --cpus-per-task="$SBATCH_CPUS_PER_TASK" --mem="$SBATCH_MEM" --time="$SBATCH_TIME" \
                run_oaml_sbatch.sh --dataset_name "$dataset" --seed "$seed" --OAML_initial "$cache" --OAML_cache "$cache" --OAML_time_budget "$time_budget" --OAML_ensemble_size "$ensemble_size"
              done
            done
          done
      fi

      # # AC: Iterate over its parameters
      if [[ "$model" == "ac" ]]; then
        for window in "${ac_exploration_window_options[@]}"; do
          for population_size in "${ac_population_size_options[@]}"; do
            echo "Submitting AC: dataset=$dataset seed=$seed window=$window population_size=$population_size"
            sbatch --partition="$SBATCH_PARTITION" --cpus-per-task="$SBATCH_CPUS_PER_TASK" --mem="$SBATCH_MEM" --time="$SBATCH_TIME" \
            run_ac_sbatch.sh --dataset_name "$dataset" --seed "$seed" --AC_exploration_window "$window" --AC_population_size "$population_size"
          done
        done
      fi

      # # ASML: Iterate over its parameters
      if [[ "$model" == "asml" ]]; then
        for window in "${asml_exploration_window_options[@]}"; do
          for ensemble_size in "${asml_ensemble_size_options[@]}"; do
            for budget in "${asml_budget_options[@]}"; do
              echo "Submitting ASML: dataset=$dataset seed=$seed window=$window ensemble_size=$ensemble_size budget=$budget"
              sbatch --partition="$SBATCH_PARTITION" --cpus-per-task="$SBATCH_CPUS_PER_TASK" --mem="$SBATCH_MEM" --time="$SBATCH_TIME" \
                run_asml_sbatch.sh --dataset_name "$dataset" --seed "$seed" --ASML_exploration_window "$window" --ASML_ensemble_size "$ensemble_size" --ASML_budget "$budget"
            done
          done
        done
      fi

      # # EAML: Iterate over its parameters
      if [[ "$model" == "eaml" ]]; then
        for population_size in "${eaml_population_size_options[@]}"; do
          for sampling_size in "${eaml_sampling_size_options[@]}"; do
            for sampling_rate in "${eaml_sampling_rate_options[@]}"; do
              echo "Submitting EAML: dataset=$dataset seed=$seed population_size=$population_size sampling_size=$sampling_size sampling_rate=$sampling_rate"
              sbatch --partition="$SBATCH_PARTITION" --cpus-per-task="$SBATCH_CPUS_PER_TASK" --mem="$SBATCH_MEM" --time="$SBATCH_TIME" \
              run_eaml_sbatch.sh --dataset_name "$dataset" --seed "$seed" --EAML_population_size "$population_size" --EAML_sampling_size "$sampling_size" --EAML_sampling_rate "$sampling_rate"
            done
          done
        done
      fi

    done
  done
done
