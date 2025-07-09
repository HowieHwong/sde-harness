import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import json
import plotly.colors
import plotly as px

def plotly_to_mpl(plotly_colorscale):
    """Convert Plotly colorscale to Matplotlib-compatible format."""
    n_colors = len(plotly_colorscale)
    mpl_colors = []
    for i in range(n_colors):
        rgb_color = plotly.colors.hex_to_rgb(plotly_colorscale[i])
        mpl_colors.append(np.array(rgb_color) / 255)  # Normalize to 0-1 range
    return mpl_colors

def calculate_hamming_distance(seq1, seq2):
    """
    Calculate the Hamming distance between two sequences.

    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence.

    Returns:
        float: Normalized Hamming distance (1 - similarity).
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)

def plot_iterative_pareto(points, pareto_points, json_data1, json_data2, fitness_dict):
    """
    Plot the Pareto frontier with points and visualize iterative progress from JSON files.

    Args:
        points (list of tuples): A list of all (x, y) points.
        pareto_points (list of tuples): The Pareto frontier points.
        json_data1 (list of list of tuples): Iterative data from the first JSON file.
        json_data2 (list of list of tuples): Iterative data from the second JSON file.
    """
    plt.figure(figsize=(16, 12))
    if pareto_points != None:
        points = np.array(points)
        pareto_points = np.array(pareto_points)
        pareto_points[:,1] = 1-pareto_points[:,1]
        pareto_points[:,0] = 1-pareto_points[:,0]
        print(pareto_points)
        # pareto_points.sort(axis=0)
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
        print(pareto_points)
        # Plot all points
        # plt.scatter(pareto_points[:, 0], pareto_points[:, 1],color=plotly.colors.qualitative.Plotly[0], alpha=0.7)

        # Highlight Pareto frontier
        plt.plot(pareto_points[:, 0], pareto_points[:, 1], marker='*',markersize=20, linewidth=10, color=plotly.colors.qualitative.Plotly[0], label="Groundtruth", alpha=0.5)

    # Plot iterative progress from JSON files
    # colors = ['blue', 'green']
    # colors = plotly.colors.sequential.
    
    # ice = px.colors.sample_colorscale(px.colors.sequential.Oryel, 15)
    # # colors = plotly.colors.qualitative.Pastel1  # Get the first color from Plotly's palette
    # colors = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in ice][5:]
    # # colors = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in colors]
    # colors1 = px.colors.sample_colorscale(px.colors.sequential.Blues, 15)
    # colors1 = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in colors1][5:]
    colors = px.colors.qualitative.Plotly[1:]
    print(colors)
    json_data = [json_data1, json_data2]
    labels = ["EA", "Ours"]
    itera = json_data1[-1]
    # print(entry)
    iteration_points = np.array([[fitness_dict[entry[1]], 1 - calculate_hamming_distance(entry[1], reference_sequence)] for entry in itera])
    print(iteration_points)
    iteration_points = iteration_points[np.argsort(iteration_points[:, 1])]
    unique_values = np.unique(iteration_points[:, 1])
    iteration_points = np.array([iteration_points[iteration_points[:, 1] == value][np.argmax(iteration_points[iteration_points[:, 1] == value][:, 0])] for value in unique_values])
    plt.plot(iteration_points[:, 1], iteration_points[:, 0], marker='^',markersize=20, linewidth=10, linestyle='-', alpha=0.5,color=colors[0], label='EA')
    
    # for i, data in enumerate(json_data):
    data = json_data[1]
    for j, iteration in enumerate(data):
        if j != len(data) - 1:
            continue
        # print([(fitness_dict[entry[1]], entry[1]) for entry in iteration[:5]])
        iteration_points = np.array([[fitness_dict[entry[1]], 1 - calculate_hamming_distance(entry[1], reference_sequence)] for entry in iteration])
        # print(iteration_points)
        iteration_points = iteration_points[np.argsort(iteration_points[:, 1])]
        
        # iteration_points.sort(axis=0)
        # print(iteration_points)
        unique_values = np.unique(iteration_points[:, 1])

        # For each unique value, find the row with the maximum value in the first column
        iteration_points = np.array([iteration_points[iteration_points[:, 1] == value][np.argmax(iteration_points[iteration_points[:, 1] == value][:, 0])] for value in unique_values])
        # plt.scatter(iteration_points[:, 1], iteration_points[:, 0], alpha=0.1+0.05*j, color=colors[i], label=f"{labels[i]}" if j == len(data) - 1 else "")

        plt.plot(iteration_points[:, 1], iteration_points[:, 0], marker='D',markersize=20, linewidth=8,linestyle='-', alpha=0.5, color=colors[1], label=f"{labels[1]}" if j == len(data) - 1 else "")
        # break
    # Title, labels, and legend
    plt.xlabel("1 - Normalized Hamming Distance",fontsize=52)
    plt.ylabel("Fitness",fontsize=52)
    plt.xticks(fontsize=48)
    plt.yticks([0.4,0.6,0.8,1.0], [0.4,0.6,0.8, 1.0], fontsize=48) 
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(fontsize=44)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./Pareto_Frontier_sum_TrpB.pdf', bbox_inches='tight')
def main(csv_file, reference_sequence, json_file1, json_file2):
    """
    Load data, calculate Pareto frontier, and visualize iterative progress.

    Args:
        csv_file (str): Path to the CSV file.
        reference_sequence (str): The reference sequence for Hamming distance calculation.
        json_file1 (str): Path to the first JSON file with iterative data.
        json_file2 (str): Path to the second JSON file with iterative data.
    """
    # Load the data
    if csv_file!= None:
        data = pd.read_csv(csv_file)
        fitness_dict = data.set_index('Combo').to_dict()['fitness']

        # Calculate Hamming distances
        data['Hamming Distance'] = data['Combo'].apply(lambda seq: calculate_hamming_distance(seq, reference_sequence))
        
        data['fitness'] = data['fitness'].apply(lambda score: 1-score)
        # # Extract points for Pareto frontier
        # print(data)
        points = list(zip(data['Hamming Distance'], data['fitness']))
    
        # score_array = np.array(points)
        # nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
        # print(nds)
        # pareto_points = score_array[nds]
        # print(pareto_points)
        # with open('./pareto_points.json', 'w') as f:
        #     json.dump(pareto_points.tolist(),f)  

        with open('./pareto_points.json', 'r') as f:
            pareto_points = json.load(f)  
    else:
        points=None
        pareto_points = None
    # Load iterative data from JSON files
    with open(json_file1, 'r') as f:
        json_data1 = json.load(f)

    with open(json_file2, 'r') as f:
        json_data2 = json.load(f)

    # Plot Pareto frontier and iterative progress
    plot_iterative_pareto(points, pareto_points, json_data1, json_data2, fitness_dict)

if __name__ == "__main__":
    # Input CSV file, reference sequence, and JSON files
    csv_file = "../single_objective/ALDE/data/TrpB/fitness.csv"
    reference_sequence = "VFVS"
    json_file1 = "./new_TrpB_fitness_scores_seed_3_48_8_384_sum_baseline.json"
    json_file2 = "./new_TrpB_fitness_scores_seed_3_48_8_384_sum_protein.json"

    main(csv_file, reference_sequence, json_file1, json_file2)

