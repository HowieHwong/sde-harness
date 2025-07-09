import json
import numpy as np

def load_json(file_name):
    """Load and return data from a JSON file."""
    with open(file_name, 'r') as f:
        return json.load(f)

def get_last_list(json_data):
    """Extract the last list from a JSON data structure."""
    return json_data[-1]

def extract_fitness_scores(data):
    """Extract fitness scores from the list of pairs."""
    return [item[0] for item in data]

def calculate_top_statistics(fitness_scores_list, top_n):
    """Get the top_n highest fitness scores from the list."""
    top_scores = np.sort(fitness_scores_list)[-top_n:]  # Get the top_n highest values
    return np.mean(top_scores)

def main(file1, file2, file3, pop_size, iteration, max_call, top_values=[1, 10, 50]):
    # Load the data from the three files
    data1 = load_json(file1)
    data2 = load_json(file2)
    data3 = load_json(file3)
    
    # Extract the last list from each data file
    last_list1 = get_last_list(data1)
    last_list2 = get_last_list(data2)
    last_list3 = get_last_list(data3)
    
    # Extract fitness scores from each last list
    fitness_scores1 = extract_fitness_scores(last_list1)
    fitness_scores2 = extract_fitness_scores(last_list2)
    fitness_scores3 = extract_fitness_scores(last_list3)
    
    # Calculate the average top1, top10, top100 for each file
    avg_top_scores = {top_n: [] for top_n in top_values}
    # print(fitness_scores1,fitness_scores2,fitness_scores3)
    for top_n in top_values:
        avg_top1 = [calculate_top_statistics(fitness_scores1, top_n), 
                    calculate_top_statistics(fitness_scores2, top_n), 
                    calculate_top_statistics(fitness_scores3, top_n)]
        avg_top_scores[top_n]=avg_top1
    # Calculate mean and std for top1, top10, and top100 scores across the three files
    if iteration == -1:
        print(f"& {pop_size}$\\times$    ", end="")
    else:
        print(f"& {pop_size}$\\times${iteration}    ", end="")
    for top_n in top_values:
        mean_top, std_top = np.mean(avg_top_scores[top_n]), np.std(avg_top_scores[top_n])
        # print(f"Top {top_n}: Mean(Std) = {mean_top:.2f}({std_top:.2f})")
        print(f"& {mean_top:.2f}$\pm${std_top:.2f}    ", end="")
    print("\\\\ &    ")  # End of the row

if __name__ == "__main__":
    # File names (replace with actual file paths if needed)
    dataset = 'Syn-3bfo'
    seed = [1,2,3]
    population_sizes = [32, 48, 96] #, 50, 100, 200, 400]
    itera = [8,8,8] #,-1,-1,-1,-1]  # Example iteration
    max_call = [384, 432, 864] #, 200, 500, 1000, 2000]
    # max_call = [160, 240, 480]
    for pop_size, iteration, max_call in zip(population_sizes, itera, max_call):
        # max_call = pop_size * iteration
        file1 = f"new_{dataset}_fitness_scores_seed_{seed[0]}_{pop_size}_{iteration}_{max_call}_protein.json"
        file2 = f"new_{dataset}_fitness_scores_seed_{seed[1]}_{pop_size}_{iteration}_{max_call}_protein.json"
        file3 = f"new_{dataset}_fitness_scores_seed_{seed[2]}_{pop_size}_{iteration}_{max_call}_protein.json"
        # file1 = f"new_{dataset}_fitness_scores_seed_{seed[0]}_{pop_size}_{iteration}_{max_call}_baseline.json"
        # file2 = f"new_{dataset}_fitness_scores_seed_{seed[1]}_{pop_size}_{iteration}_{max_call}_baseline.json"
        # file3 = f"new_{dataset}_fitness_scores_seed_{seed[2]}_{pop_size}_{iteration}_{max_call}_baseline.json"
        
        main(file1, file2, file3, pop_size, iteration, max_call)
