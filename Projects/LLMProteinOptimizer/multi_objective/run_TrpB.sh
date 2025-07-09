#!/bin/bash

# Define the arrays with 7 values each
population_size=(48 48) # 50 100 200 400)
max_oracle_call=(-1 -1 -1) # 200 500 1000 2000)
iteration=(8 8 8) # -1 -1 -1 -1)
model=(molleo_multi molleo_multi_pareto)

length=${#population_size[@]}


for i in $(seq 0 $((length - 1))); do
    pop_size=${population_size[$i]}
    max_call=${max_oracle_call[$i]}
    iter=${iteration[$i]}
    m=${model[$i]}
    # Perform your operation here
    python run.py $m --mol_lm Llama3 --dataset Syn-3bfo --population_size $pop_size --iteration $iter --max_oracle_calls $max_call --seed 1 2 3 
    
done
