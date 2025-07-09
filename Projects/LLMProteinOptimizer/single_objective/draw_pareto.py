import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from run import Oracle_model
import json
def calculate_hamming_distance(seq1, seq2):
    """
    Calculate the Hamming distance between two sequences.
    
    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence.
    
    Returns:
        int: Hamming distance.
    """
    # print(seq1, seq2)
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    return 1 - (sum(c1 != c2 for c1, c2 in zip(seq1, seq2))/len(seq1))

def pareto_frontier(points, maximize=True):
    """
    Calculate the Pareto frontier from a set of points.
    
    Args:
        points (list of tuples): A list of (x, y) points.
        maximize (bool): True if the goal is to maximize values, False if minimizing.
        
    Returns:
        list of tuples: The points on the Pareto frontier.
    """
    score_array = np.array(points)
    nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
    print(len(nds))
    points = sorted(points, key=lambda x: x[0], reverse=maximize)
    pareto_front = [points[0]]  # Initialize with the first point

    for point in points[1:]:
        if (maximize and point[1] >= pareto_front[-1][1]) or (not maximize and point[1] <= pareto_front[-1][1]):
            pareto_front.append(point)
    
    return pareto_front

def select_replaxed_pareto_front(points, pareto_points):

    distances = np.linalg.norm(pareto_points[:, np.newaxis] - points, axis=2)
    closest_indices = np.argmin(distances, axis=1)
    relaxed_pareto = points[closest_indices]
    unique_relaxed_pareto = np.unique(relaxed_pareto, axis=0)

    return unique_relaxed_pareto
def plot_pareto(points, pareto_points, picked_points):
    """
    Plot the points and Pareto frontier.
    
    Args:
        points (list of tuples): A list of all (x, y) points.
        pareto_points (list of tuples): The Pareto frontier points.
    """
    points = np.array(points)
    pareto_points = np.array(pareto_points)
    picked_points = np.array(picked_points)
    # relaxed_points = select_replaxed_pareto_front(points, pareto_points)
    print(len(pareto_points))
    # plt.scatter(points[:, 0], points[:, 1], label="All Points", alpha=0.5)

    # Highlight Pareto frontier
    plt.plot(pareto_points[:, 0], pareto_points[:, 1], color='red', label="Pareto Frontier", linewidth=2, alpha=0.5)
    # plt.scatter(pareto_points[:, 0], pareto_points[:, 1], color='red', zorder=5)
    # plt.plot(relaxed_points[:, 0], relaxed_points[:, 1], color='blue', label="Relaxed Pareto Frontier", linewidth=2)
    # plt.scatter(relaxed_points[:, 0], relaxed_points[:, 1], color='blue', zorder=5)
    # plt.plot(picked_points[:, 0], picked_points[:, 1], color='green', label="Proposed Pareto Frontier", linewidth=2)
    plt.scatter(picked_points[:, 0], picked_points[:, 1], color='green', zorder=5)

    # Annotate specific points (e.g., A, B, C)
    # if labels:
    #     for (x, y), label in zip(points, labels):
    #         plt.text(x, y, label, fontsize=10, ha='right')

    # Title, labels, and legend
    plt.title("Pareto Frontier Visualization")
    plt.xlabel("Hamming Distance")
    plt.ylabel("Fitness")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.axhline(color="gray", linestyle="--", alpha=0.5)
    # plt.axvline(color="gray", linestyle="--", alpha=0.5)
    plt.legend()
    # plt.grid(True)
    plt.savefig('./Pareto_Frontier_TrpB.png')

def main(csv_file, reference_sequence,oracle,dataset):
    """
    Load data from a CSV, calculate Hamming distances, and plot the Pareto frontier.
    
    Args:
        csv_file (str): Path to the CSV file.
        reference_sequence (str): The reference sequence for Hamming distance calculation.
    """
    # Load the data
    
    data = pd.read_csv(csv_file)
    # for seed in [1,2,3]:
    if dataset =='AAV':
        with open('/home/jhe/Molleo_Protein/single_objective/AAV_Llama3fitness_scores_seed_1_32_8_288_protein.json','r') as f:
            picked = json.load(f)
    else:
        with open('/home/jhe/Molleo_Protein/single_objective/GFP_EAfitness_scores_seed_1_96_8_864_protein.json','r') as f:
            picked = json.load(f)
    # print(len(picked))
    # exit()
    # for p in picked[-1]:
    #     print(len(p))
    #     print(p[-1],p[-2])
    #     exit()
    # exit()

    picked_hamming = [calculate_hamming_distance(p[1], reference_sequence) for p in picked[-1]]
    # picked_fitness = [data[data['Combo'] == seq[1]]['fitness'].to_list()[0] for seq in picked[-1]]
    picked_fitness = [p[0] for p in picked[-1]]

    # result = data[data['sequence'] == reference_sequence]['score']
    # print(result.mean())
    # print(result.min(),result.max())

    # percentile = (data['score'] <= result.max()).sum() / len(data['score']) * 100
    # print(f"orginal dataset percentile of wt:{percentile}")
    # # Calculate Hamming distances
    data['Hamming Distance'] = data['sequence'].apply(lambda seq: calculate_hamming_distance(seq, reference_sequence))
    # # data["normalize"]
    data['ML_fitness'] = data['sequence'].apply(lambda seq: oracle(seq))
    data['ML_fitness'] = data['ML_fitness'].apply(lambda x: x.item())
    print(oracle(reference_sequence))
    # percentile = (data['ML_fitness'] <= oracle(reference_sequence)).sum() / len(data['ML_fitness']) * 100
    # print(percentile)
    print("***************")
    print(data['ML_fitness'].max(),data['ML_fitness'].min())
    print(data['ML_fitness'].mean(),data['ML_fitness'].std())
    print("*********")
    print(max(picked_fitness),min(picked_fitness))
    print(np.mean(picked_fitness))
    exit()

    # exit()
    # print(data['ML_fitness'])
    # exit()
    # population_mol = [s for s in data[:]]
    # data_ML_fitness = oracle([mol for mol in population_mol])
    # .sort_values(by='score')['sequence'].tolist()
    # Extract points for Pareto frontier
    # points = list(zip(data['Hamming Distance'], data['fitness']))
    #

    points = list(zip(data['Hamming Distance'],data['ML_fitness']))
    # points = list(zip(picked_hamming,picked_fitness))
    picked_points = list(set(list(zip(picked_hamming, picked_fitness))))
    # Calculate Pareto frontier
    pareto_point_path = f'/home/jhe/Molleo_Protein/single_objective/{dataset}_pareto_points.json'
    if os.path.exists(pareto_point_path):
        with open(pareto_point_path, 'r') as f:
            pareto_points = json.load(f)  
    else:
        pareto_points = pareto_frontier(points, maximize=True)
        with open(pareto_point_path, 'w') as f:
            json.dump(pareto_points, f)

    # Plot Pareto frontier
    print(pareto_points, picked_points)
    plot_pareto(points, pareto_points, picked_points)

if __name__ == "__main__":
    # Input CSV file and reference sequence
    dataset = 'GFP'
    oracle = Oracle_model(dataset=dataset)
    csv_file = f"/home/jhe/Molleo_Protein/data/{dataset}/gt_medium_range.csv"  # Replace with your file path
    # csv_file = f"/home/jhe/Molleo_Protein/data/{dataset}/ground_truth.csv"  
    # reference_sequence = "VFVS"  # Replace with your reference sequence
    reference_sequence = pd.read_csv(f'/home/jhe/Molleo_Protein/data/{dataset}/{dataset}_wild_type.csv').wild_type_sequence.iloc[0]
    # rs_AAV = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
    # rs_GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    # # print(len(reference_sequence))
    # # print(reference_sequence)
    # print(rs_AAV[560:588])
    # print(len(rs_AAV))
    # exit()
    # print(rs_GFP)
    # print(reference_sequence)

    # exit()
    main(csv_file= csv_file, reference_sequence = reference_sequence,oracle=oracle,dataset=dataset)