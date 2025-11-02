models=("none" "openai/gpt-4o")
datasets=("syn-3bfo" "gb1" "trpb" "aav" "gfp")
tasks=("multi" "single")

# Example: get a length if you need it
# length=${#datasets[@]}

for m in "${models[@]}"; do
  for d in "${datasets[@]}"; do
    for t in "${tasks[@]}"; do
      echo "Running: python3 cli.py $t --oracle $d --generations 8 --population-size 200 --offspring-size 100 --model $m"
      python3 cli.py "$t" --oracle "$d" --generations 8 --population-size 200 --offspring-size 100 --model "$m" & 
    done
  done
done