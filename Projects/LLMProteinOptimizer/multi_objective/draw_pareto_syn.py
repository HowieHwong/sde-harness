from matplotlib.patches import bbox_artist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import json
import plotly.colors
import plotly as px
import potts_model

ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
A2N = {a: n for n, a in enumerate(ALPHABET)}
A2N["X"] = 20

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

def plot_iterative_pareto(json_data1, json_data2):
    """
    Plot the Pareto frontier with points and visualize iterative progress from JSON files.

    Args:
        points (list of tuples): A list of all (x, y) points.
        pareto_points (list of tuples): The Pareto frontier points.
        json_data1 (list of list of tuples): Iterative data from the first JSON file.
        json_data2 (list of list of tuples): Iterative data from the second JSON file.
    """
    plt.figure(figsize=(16, 12))
    reference_sequence = 'SKLQICVEPTSQKLMPGSTLVLQCVAVGSPIPHYQWFKNELPLTHETKKLYMVPYVDLEHQGTYWCHVYNDRDSQDSKKVEIIID'
    landscape = potts_model.load_from_mogwai_npz(f'data/3bfo_1_A_model_state_dict.npz', coupling_scale=1.0)
    colors = px.colors.qualitative.Plotly[1:]
    # print(colors)
    json_data = [json_data1, json_data2]
    labels = ["EA", "Ours"]
    itera = json_data1[-1]
    # print(json_data)
    # print(entry)
    iteration_points = np.array([[landscape.evaluate([np.array([A2N[i] for i in entry[1]])])[0], 1 - calculate_hamming_distance(entry[1], reference_sequence)] for entry in itera])
    # print(iteration_points)
    iteration_points = iteration_points[np.argsort(iteration_points[:, 1])]
    unique_values = np.unique(iteration_points[:, 1])
    iteration_points = np.array([iteration_points[iteration_points[:, 1] == value][np.argmax(iteration_points[iteration_points[:, 1] == value][:, 0])] for value in unique_values])
    plt.plot(iteration_points[:, 1], iteration_points[:, 0], marker='^',markersize=20, linewidth=10,linestyle='-', alpha=0.5,color=colors[0], label='EA')
    
    # for i, data in enumerate(json_data):
    data = json_data[1]
    for j, iteration in enumerate(data):
        if j != len(data) - 1:
            continue
        # print([(fitness_dict[entry[1]], entry[1]) for entry in iteration[:5]])
        iteration_points = np.array([[landscape.evaluate([np.array([A2N[i] for i in entry[1]])])[0], 1 - calculate_hamming_distance(entry[1], reference_sequence)] for entry in iteration])
        # print(iteration_points)
        # print(iteration_points)
        iteration_points = iteration_points[np.argsort(iteration_points[:, 1])]
        
        # iteration_points.sort(axis=0)
        # print(iteration_points)
        unique_values = np.unique(iteration_points[:, 1])

        # For each unique value, find the row with the maximum value in the first column
        iteration_points = np.array([iteration_points[iteration_points[:, 1] == value][np.argmax(iteration_points[iteration_points[:, 1] == value][:, 0])] for value in unique_values])
        # plt.scatter(iteration_points[:, 1], iteration_points[:, 0], alpha=0.1+0.05*j, color=colors[i], label=f"{labels[i]}" if j == len(data) - 1 else "")

        plt.plot(iteration_points[:, 1], iteration_points[:, 0], marker='D', markersize=20, linewidth=10, linestyle='-', alpha=0.5, color=colors[1], label=f"{labels[1]}" if j == len(data) - 1 else "")
        # break
    # Title, labels, and legend
    # plt.title("Pareto Frontier and Iterative Progress Visualization")
    
    plt.xlabel("1 - Normalized Hamming Distance",fontsize=52)
    plt.ylabel("Fitness",fontsize=52)
    plt.xticks([0.96,0.97,0.98,0.99,1.0],[0.96,0.97,0.98,0.99,1.0],fontsize=48)
    plt.yticks([-0.5, 0.0, 1.0, 2.0, 3.0], ['','0.0', '1.0', '2.0', '3.0'], fontsize=48) 
    plt.locator_params(axis='x', nbins=5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(fontsize=44)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./Pareto_Frontier_Syn-3bfo.pdf', bbox_inches='tight')

def main(json_file1, json_file2):
    """
    Load data, calculate Pareto frontier, and visualize iterative progress.

    Args:
        csv_file (str): Path to the CSV file.
        reference_sequence (str): The reference sequence for Hamming distance calculation.
        json_file1 (str): Path to the first JSON file with iterative data.
        json_file2 (str): Path to the second JSON file with iterative data.
    """
    # Load iterative data from JSON files
    with open(json_file1, 'r') as f:
        json_data1 = json.load(f)

    with open(json_file2, 'r') as f:
        json_data2 = json.load(f)

    # Plot Pareto frontier and iterative progress
    plot_iterative_pareto(json_data1, json_data2)

if __name__ == "__main__":
    json_file1 = "./new_Syn-3bfo_fitness_scores_seed_1_48_8_384_baseline.json"
    json_file2 = "./new_Syn-3bfo_fitness_scores_seed_1_48_8_384_protein.json"
    # json_file1 = "../single_objective/new_Syn-3bfo_fitness_scores_seed_1_48_8_432_K_10_baseline.json"
    # json_file2 = "../single_objective/new_Syn-3bfo_fitness_scores_seed_1_48_8_432_K_10_protein.json"
   
    main(json_file1, json_file2)

