#!/bin/bash

# Define the arrays with 7 values each
population_size=(32)
max_oracle_call=(-1 -1 -1)
iteration=(4 4 4)

length=${#population_size[@]}

# Iterate through the arrays together
for i in $(seq 0 $((length - 1))); do
    pop_size=${population_size[$i]}
    max_call=${max_oracle_call[$i]}
    iter=${iteration[$i]}

    # Perform your operation here
    python run.py molleo --mol_lm Llama3 --dataset GB1 --population_size $pop_size --iteration $iter --max_oracle_calls $max_call --seed 1 2 3 --oracles fitness 
    
    # Example of running a command
    # ./your_script.sh --population_size $pop_size --max_oracle_call $max_call --iteration $iter
done
