import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from the file
df = pd.read_csv('./ALDE/data/TrpB/fitness.csv')

# Extract sequences and their fitness scores
sequences = df['Combo']
characters = sorted(set(char for seq in sequences for char in seq))

# Dictionary to store fitness for each category
results = {"First Two": {}, "Last Two": {}, "Last Three": {}, "Full Sequence": {}}

# Iterate through the entire sequence space
for index, row in df.iterrows():
    combo = row['Combo']
    fitness = row['fitness']
    
    # Extract combinations
    first_two = combo[:2]
    last_two = combo[-2:]
    last_three = combo[-3:]
    full_seq = combo

    # Update results
    results["First Two"][first_two] = results["First Two"].get(first_two, []) + [fitness]
    results["Last Two"][last_two] = results["Last Two"].get(last_two, []) + [fitness]
    results["Last Three"][last_three] = results["Last Three"].get(last_three, []) + [fitness]
    results["Full Sequence"][full_seq] = results["Full Sequence"].get(full_seq, []) + [fitness]

# Calculate mean fitness for each combination
for key in results.keys():
    results[key] = {k: np.mean(v) for k, v in results[key].items()}

# Convert to DataFrames and select top 20 combinations
first_two_df = sorted(results["First Two"].items(), key=lambda x: x[1], reverse=True)[:20]
last_two_df = sorted(results["Last Two"].items(), key=lambda x: x[1], reverse=True)[:20]
last_three_df = sorted(results["Last Three"].items(), key=lambda x: x[1], reverse=True)[:20]
full_seq_df = sorted(results["Full Sequence"].items(), key=lambda x: x[1], reverse=True)[:20]

# Prepare data for heatmaps
categories = ["First Two Sites", "Last Two Sites", "Last Three Sites", "Full Sequence"]
dataframes = [
    pd.DataFrame(first_two_df, columns=["Combination", "Fitness"]).set_index("Combination"),
    pd.DataFrame(last_two_df, columns=["Combination", "Fitness"]).set_index("Combination"),
    pd.DataFrame(last_three_df, columns=["Combination", "Fitness"]).set_index("Combination"),
    pd.DataFrame(full_seq_df, columns=["Combination", "Fitness"]).set_index("Combination")
]
# Plot the heatmaps
fig, axes = plt.subplots(1, 4, figsize=(28, 18), constrained_layout=True)

for ax, category, h_data in zip(axes, categories, dataframes):
    sns.heatmap(
        h_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=False,  # Add colorbar only for the last heatmap
        yticklabels=True,
        xticklabels=False,
        ax=ax,
        annot_kws={"size": 40},
        # cbar_kws={"size": 44}
    )
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=44)

    plt.rcParams['axes.labelsize'] = 44
    plt.rcParams['axes.titlesize'] = 44
    ax.set_title(category)
    ax.set_ylabel("Combination" if category == "First Two" else "")
    ax.set_xlabel("")
    ax.set_yticklabels(h_data.index, rotation=0, ha="right",fontsize=48)
    # ax.set_yticklabels(fontsize=48)
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="center")
# ax.yaxis.set_tick_params(labelrotation=0)
# Add a shared title
# plt.tight_layout()
# plt.suptitle("TrpB", fontsize=16)
plt.savefig('./TrpB_heatmaps.png')
plt.show()
