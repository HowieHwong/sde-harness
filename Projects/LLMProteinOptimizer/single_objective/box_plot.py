import json
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_boxplot(json_file_ours, json_file_baseline, output_path):
    # Load data from the JSON files
    with open(json_file_ours, 'r') as f:
        ours_data = json.load(f)
    
    with open(json_file_baseline, 'r') as f:
        baseline_data = json.load(f)

    # Extract only the scores (first element of each inner list)
    def extract_scores(data):
        return [[score[0] for score in iteration] for iteration in data]

    ours_scores = extract_scores(ours_data)
    baseline_scores = extract_scores(baseline_data)

    # Prepare iterations for x-axis
    iterations = [i for i in range(len(ours_scores))]
    x_positions = np.arange(len(iterations))

    # Create the plot
    plt.figure(figsize=(14, 12))

    # Plot "Ours" data
    bplot1 = plt.boxplot(
        ours_scores, 
        positions=x_positions - 0.18,  # Shift left
        widths=0.35, patch_artist=True, 
        boxprops=dict(facecolor='blue', alpha=0.5),
        medianprops=dict(color='black'), 
        flierprops=dict(markerfacecolor='blue', markeredgecolor='blue', alpha=0.5),
        whis=np.inf
    )

    # Plot "Baseline EA" data
    bplot2 = plt.boxplot(
        baseline_scores, 
        positions=x_positions + 0.18,  # Shift right
        widths=0.35, patch_artist=True, 
        boxprops=dict(facecolor='orange', alpha=0.5),
        medianprops=dict(color='black'), 
        flierprops=dict(markerfacecolor='orange', markeredgecolor='orange', alpha=0.5),
        whis=np.inf
    )

    # Adjust x-axis
    plt.xticks(ticks=x_positions, labels=iterations, fontsize=48)
    plt.yticks(fontsize=48) 
    # Add grid, title, labels, and legend
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.title("Comparison of Fitness Scores Across Iterations", fontsize=48)
    plt.xlabel("Iterations", fontsize=52)
    plt.ylabel("Fitness Score", fontsize=52)

    # Add a legend
    plt.legend(
        [bplot1["boxes"][0], bplot2["boxes"][0]], 
        ["Ours", "EA"],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        fontsize=44,
        ncol=2
    )
    # ax = plt.gca()  # Get current axes
    # ax.get_legend().remove()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Example usage
dataset = 'GFP'
for seed in [1,2,3]:
    plot_comparison_boxplot(f"./new_{dataset}_fitness_scores_seed_{seed}_96_8_864_protein.json", f"./new_{dataset}_fitness_scores_seed_{seed}_96_8_864_baseline.json", f"{dataset}_box_96_seed{seed}.png")
