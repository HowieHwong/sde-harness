"""Analysis mode for evaluating results."""
import os
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def find_results_files(data: str, model: str) -> List[str]:
    """Find all final_results.json files for given dataset and model."""
    # Search patterns for different directory structures
    patterns = [
        f"*/{model}_{data}/*/final_results.json",
        f"*/*/{model}_{data}/*/final_results.json",
        f"{model}_{data}/*/final_results.json",
        # Handle cases where model name might be in path
        f"*/openai/*{data}/*/final_results.json",
        f"*/anthropic/*{data}/*/final_results.json",
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    found_files = sorted(list(set(found_files)))
    return found_files


def analyze_final_results(json_path: str) -> Dict[str, Any]:
    """Analyze a single final_results.json file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract key metrics
    aggregate = data.get('aggregate', {})
    result = {
        'path': json_path,
        'num_rounds': aggregate.get('num_rounds', len(data.get('round_results', []))),
        'mean_hit_rate': aggregate.get('mean_hit_rate', 0),
        'std_hit_rate': aggregate.get('std_hit_rate', 0),
        'total_unique_genes': aggregate.get('total_unique_genes', 0),
        'total_hits': data.get('total_hits', 0),
        'hit_rates_by_round': aggregate.get('hit_rates_by_round', []),
        'hits_progression': data.get('hits_progression', [])
    }
    
    # Add individual hit genes info if available
    if 'hit_genes_with_scores' in data:
        result['hit_genes'] = [(h['gene'], h['score']) for h in data['hit_genes_with_scores']]
    
    return result


def calc_stats(data: str, model: str, rounds: int, trials: int, essential: bool) -> Dict[str, Any]:
    """Calculate statistics from final_results.json files."""
    # Find all relevant final_results.json files
    results_files = find_results_files(data, model)
    
    if not results_files:
        print(f"ERROR: No final_results.json files found for {model} on {data}")
        print("Searched for patterns like:")
        print(f"  - */{model}_{data}/*/final_results.json")
        print(f"  - */openai/*{data}/*/final_results.json")
        return {
            "model": model,
            "dataset": data,
            "error": "No results files found"
        }
    
    print(f"\nFound {len(results_files)} result file(s):")
    all_results = []
    
    for json_path in results_files:
        print(f"\nAnalyzing: {json_path}")
        result = analyze_final_results(json_path)
        all_results.append(result)
        
        # Print individual result
        print(f"  Rounds: {result['num_rounds']}")
        print(f"  Mean hit rate: {result['mean_hit_rate']:.4f} ± {result['std_hit_rate']:.4f}")
        print(f"  Total unique genes: {result['total_unique_genes']}")
        print(f"  Total hits: {result['total_hits']}")
        print(f"  Hits progression: {result['hits_progression']}")
        print(f"  Hit rates by round: {[f'{hr:.3f}' for hr in result['hit_rates_by_round']]}")
        
        # Show top hits if available
        if 'hit_genes' in result and result['hit_genes']:
            print(f"  Top hit genes:")
            # Sort by absolute score
            sorted_hits = sorted(result['hit_genes'], key=lambda x: abs(x[1]), reverse=True)
            for gene, score in sorted_hits[:5]:
                print(f"    - {gene}: {score:.4f}")
            if len(sorted_hits) > 5:
                print(f"    ... and {len(sorted_hits)-5} more")
    
    # Aggregate across all trials
    if len(all_results) > 1:
        mean_hit_rates = [r['mean_hit_rate'] for r in all_results]
        total_hits = [r['total_hits'] for r in all_results]
        
        print(f"\n=== AGGREGATE ACROSS {len(all_results)} TRIALS ===")
        print(f"Mean hit rate: {np.mean(mean_hit_rates):.4f} ± {np.std(mean_hit_rates):.4f}")
        print(f"Mean total hits: {np.mean(total_hits):.2f} ± {np.std(total_hits):.2f}")
    
    return {
        "model": model,
        "dataset": data,
        "trials_found": len(all_results),
        "results": all_results
    }


def run_analyze(args) -> Dict[str, Any]:
    """Run analysis on results."""
    print(f"\nAnalyzing results for {args.model} on {args.dataset}")
    print("="*60)
    
    results = calc_stats(
        args.dataset,
        args.model,
        args.rounds,
        args.trials,
        args.essential
    )
    
    return results