#!/usr/bin/env bash
# Reproduce every single-objective run reported in the paper.
# Usage:  bash run_all.sh
#
# Each (model, dataset) pair is launched as a background job and writes
#   results/results_single_<dataset>_<seed>_<model>.json
#   logs/<dataset>_<task>_<model>.log
# Edit `models`, `datasets`, `seeds` or the GA settings below as needed.

set -u

# "none" runs the GA without LLM-guided mutations -> saved as the `baseline` row.
models=("none" "openai/gpt-5-mini" "deepseek/deepseek-reasoner" "anthropic/claude-sonnet-4-5" "openai/gpt-5" "openai/gpt-5-chat-latest")
datasets=("syn-3bfo" "gb1" "trpb" "aav" "gfp")
tasks=("single")
seeds=(0)

GENERATIONS=8
POPULATION_SIZE=200
OFFSPRING_SIZE=100

mkdir -p logs results

for m in "${models[@]}"; do
  for d in "${datasets[@]}"; do
    for t in "${tasks[@]}"; do
      for s in "${seeds[@]}"; do
        tag="${m##*/}"
        echo "Running: python cli.py $t --oracle $d --generations $GENERATIONS --population-size $POPULATION_SIZE --offspring-size $OFFSPRING_SIZE --seed $s --model $m"
        nohup python -u cli.py "$t" \
          --oracle "$d" \
          --generations "$GENERATIONS" \
          --population-size "$POPULATION_SIZE" \
          --offspring-size "$OFFSPRING_SIZE" \
          --seed "$s" \
          --model "$m" \
          > "logs/${d}_${t}_${tag}_seed${s}.log" 2>&1 &
      done
    done
  done
done

wait
echo "All runs finished. Summarize with:"
echo "  python src/analyze.py --glob './results/results_single_*.json' --higher-is-better 1"
