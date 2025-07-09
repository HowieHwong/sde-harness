import pandas as pd
import random
import numpy as np

ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")

def load_data(csv_file):
    """Load sequence data from a CSV file."""
    data = pd.read_csv(csv_file)
    if 'Combo' not in data.columns or 'fitness' not in data.columns:
        raise ValueError("CSV must contain 'Combo' and 'fitness' columns.")
    return data

def random_mutation(sequence, valid_chars):
    """Perform a single random mutation on a sequence."""
    seq_list = list(sequence)
    # Pick a random position to mutate
    mutate_pos = random.randint(0, len(seq_list) - 1)
    # Ensure the new character is different
    new_char = random.choice(valid_chars)
    while new_char == seq_list[mutate_pos]:
        new_char = random.choice(valid_chars)
    seq_list[mutate_pos] = new_char
    return ''.join(seq_list)

def find_fitness(sequence, data):
    """Find the fitness score of a sequence from the dataset."""
    match = data[data['Combo'] == sequence]
    if not match.empty:
        return match['fitness'].iloc[0]
    return 0  # Return NaN if sequence not found

def baseline_mutation(csv_file, output_file, num_mutations=1000):
    """Perform random mutations on sequences, log average fitness, and save the results."""
    data = load_data(csv_file)
    mutated_data = []
    total_original_fitness = 0
    total_mutated_fitness = 0
    valid_mutated_count = 0  # Count of valid mutations (with non-NaN fitness)

    for _ in range(num_mutations):
        # Randomly pick a sequence from the data
        row = data.sample(n=1).iloc[0]
        original_sequence = row['Combo']
        original_fitness = row['fitness']
        
        # Perform random mutation
        mutated_sequence = random_mutation(original_sequence, ALL_AAS)
        
        # Find fitness for the mutated sequence
        mutated_fitness = find_fitness(mutated_sequence, data)
        
        # Update totals for averaging
        total_original_fitness += original_fitness
        if not np.isnan(mutated_fitness):
            total_mutated_fitness += mutated_fitness
            valid_mutated_count += 1
        
        mutated_data.append({
            'Original_Combo': original_sequence,
            'Mutated_Combo': mutated_sequence,
            'Original_Fitness': original_fitness,
            'Mutated_Fitness': mutated_fitness
        })

    # Calculate averages
    avg_original_fitness = total_original_fitness / num_mutations
    avg_mutated_fitness = (
        total_mutated_fitness / valid_mutated_count if valid_mutated_count > 0 else np.nan
    )
    
    # Log averages
    print(f"Average Original Fitness: {avg_original_fitness:.4f}")
    print(f"Average Mutated Fitness: {avg_mutated_fitness:.4f}")

    # Convert to DataFrame and save results
    mutated_df = pd.DataFrame(mutated_data)
    mutated_df.to_csv(output_file, index=False)
    print(f"Mutated data saved to {output_file}")

# Define parameters
csv_file = '/cluster/tufts/liulab/yiwan01/MOLLEO/single_objective/ALDE/data/GB1/fitness.csv'  # Input CSV file with 'Combo' and 'fitness'
output_file = 'mutated_data_baseline.csv'  # Output CSV file

# Perform baseline mutation
baseline_mutation(csv_file, output_file, num_mutations=20000)
