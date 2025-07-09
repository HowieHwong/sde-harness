#!/bin/bash

# Define the arrays with 7 values each
population_size=(32 48 96) # 50 100 200 400)
max_oracle_call=(-1 -1 -1) # 200 500 1000 2000)
iteration=(8 8 8) # -1 -1 -1 -1)

length=${#population_size[@]}

# # Iterate through the arrays together
# for i in $(seq 0 $((length - 1))); do
#     pop_size=${population_size[$i]}
#     max_call=${max_oracle_call[$i]}
#     iter=${iteration[$i]}

#     # Perform your operation here
#     python run.py molleo --mol_lm EA --dataset TrpB --population_size $pop_size --iteration $iter --max_oracle_calls $max_call --seed 1 2 3 --oracles fitness 
    
#     # Example of running a command
#     # ./your_script.sh --population_size $pop_size --max_oracle_call $max_call --iteration $iter
# done

for i in $(seq 0 $((length - 1))); do
    pop_size=${population_size[$i]}
    max_call=${max_oracle_call[$i]}
    iter=${iteration[$i]}

    # Perform your operation here
    python run.py molleo --mol_lm Llama3 --dataset TrpB --population_size $pop_size --iteration $iter --max_oracle_calls $max_call --seed 1 2 3 --oracles fitness
    
done
