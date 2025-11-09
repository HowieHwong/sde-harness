#!/bin/bash

#models=("gpt-4o" "gpt-5-mini") # TODO: Which other ones?
models=("gpt-4o")
datasets=("uspto-190" "pistachio-hard")
max_oracle_calls=(100)

echo "Starting benchmark runs..."
echo "Models: ${models[@]}"
echo "Datasets: ${datasets[@]}"
echo "Max oracle calls: ${max_oracle_calls[@]}"
echo "Note the runs will be *sequential*"

start_time=$(date +%s.%N)

for m in "${models[@]}"; do
  for d in "${datasets[@]}"; do
    for o in "${max_oracle_calls[@]}"; do
      echo "Running: python cli.py --dataset $d --model $m --max_oracle_calls $o"
      python cli.py --dataset "$d" --model "$m" --max_oracle_calls "$o"
    done
  done
done

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
elapsed_rounded=$(printf "%.2f" $elapsed)

echo ""
echo "Finished all runs in ${elapsed_rounded} seconds"