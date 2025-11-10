from typing import Dict, List, Tuple
import argparse
import os
import csv


# Hard-coded non-LLM-based model values from https://openreview.net/forum?id=NhkNX8jYld&noteId=9wCQSd8Tfu
BASELINE_RESULTS = {
    "Graph2Edits(MCTS)": {
        "USPTO Easy": 90.0,
        "USPTO-190": 42.7,
        "Pistachio Reachable": 77.3,
        "Pistachio Hard": 26.0
    },
    "RootAligned(MCTS)": {
        "USPTO Easy": 98.0,
        "USPTO-190": 79.4,
        "Pistachio Reachable": 99.3,
        "Pistachio Hard": 83.0
    },
    "LocalRetro(MCTS)": {
        "USPTO Easy": 92.5,
        "USPTO-190": 44.3,
        "Pistachio Reachable": 86.7,
        "Pistachio Hard": 52.0
    },
    "Graph2Edits(Retro*)": {
        "USPTO Easy": 92.0,
        "USPTO-190": 51.1,
        "Pistachio Reachable": 94.0,
        "Pistachio Hard": 71.0
    },
    "RootAligned(Retro*)": {
        "USPTO Easy": 99.0,
        "USPTO-190": 86.8,
        "Pistachio Reachable": 98.7,
        "Pistachio Hard": 78.0
    },
    "LocalRetro(Retro*)": {
        "USPTO Easy": 95.5,
        "USPTO-190": 51.0,
        "Pistachio Reachable": 97.3,
        "Pistachio Hard": 63.0
    }
}

def get_dataset_name_and_size(dataset: str) -> Tuple[str, int]:
    """Get dataset name and size."""
    mapping = {
        "uspto-easy": ("USPTO Easy", 200),
        "uspto-190": ("USPTO-190", 190),
        "pistachio-reachable": ("Pistachio Reachable", 100),
        "pistachio-hard": ("Pistachio Hard", 100)
    }
    try: return mapping[dataset]
    except KeyError: raise ValueError(f"Invalid dataset: {dataset}")

def crawl_results_directory(results_dir: str) -> Dict[str, Dict[str, float]]:
    """Crawl the results directory and extract solve rates."""
    llm_results = {}
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    for model_name in model_dirs:
        model_path = os.path.join(results_dir, model_name)
        algorithm_name = f"LLM-Syn-Planner({model_name})"
        llm_results[algorithm_name] = {}
        
        # Get all dataset directories for this model
        dataset_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        
        for dataset in dataset_dirs:
            dataset_path = os.path.join(model_path, dataset)
            dataset_display_name, dataset_size = get_dataset_name_and_size(dataset)
            
            # Get results for N=100 only
            oracle_path = os.path.join(dataset_path, "100")
            
            if os.path.exists(oracle_path):
                if not os.path.exists(os.path.join(oracle_path, "solved_routes")): 
                    print(f"{oracle_path} contains no solved routes, setting solve rate to 0")
                    solve_rate = 0.0
                else:
                    solve_rate = round(len(os.listdir(os.path.join(oracle_path, "solved_routes"))) / dataset_size * 100, 1)
            else:
                print(f"{oracle_path} does not exist (likely experimental configuration was not run), setting solve rate to None")
                solve_rate = None
            
            llm_results[algorithm_name][dataset_display_name] = solve_rate
    
    return llm_results

def write_benchmark_table(output_file: str, baseline_results: Dict, llm_results: Dict) -> None:
    """Write the benchmark results table to a CSV file."""
    # Combine baseline and LLM results
    all_results = {**baseline_results, **llm_results}
    
    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header row
        header_row = ["Algorithm", "USPTO Easy", "USPTO-190", "Pistachio Reachable", "Pistachio Hard"]
        csv_writer.writerow(header_row)
        
        # Write data rows
        benchmarks = ["USPTO Easy", "USPTO-190", "Pistachio Reachable", "Pistachio Hard"]
        for algorithm, results in all_results.items():
            row = [algorithm]
            for benchmark in benchmarks:
                if benchmark in results:
                    row.append(results[benchmark])
                else:
                    row.append(None)
            csv_writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark results table")
    parser.add_argument("--results_dir", type=str, default="./synplanner_results", help="Path to results directory")
    parser.add_argument("--output_file", type=str, default="benchmark_results_table.csv", help="Output CSV file name")
    args = parser.parse_args()
    
    assert os.path.exists(args.results_dir), f"Results directory: {args.results_dir} does not exist."
    
    # Crawl results directory to get LLM results
    llm_results = crawl_results_directory(args.results_dir)
    
    if llm_results: 
        write_benchmark_table(args.output_file, BASELINE_RESULTS, llm_results)
        print(f"Benchmark table saved to {args.output_file}")
    else: 
        print(f"No LLM results found in {args.results_dir}. No benchmark table will be created.")
    
if __name__ == "__main__":
    main()
