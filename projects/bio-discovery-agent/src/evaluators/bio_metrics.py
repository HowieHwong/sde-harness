"""Biological evaluation metrics for BioDiscoveryAgent."""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import os


def calculate_hit_rate(predicted_genes: List[str], 
                      ground_truth_genes: List[str],
                      essential_genes: Optional[List[str]] = None) -> float:
    """
    Calculate hit rate for predicted genes.
    
    Args:
        predicted_genes: List of predicted gene names
        ground_truth_genes: List of ground truth gene names
        essential_genes: Optional list of essential genes to exclude
        
    Returns:
        Hit rate as a float between 0 and 1
    """
    # Convert to sets for efficient intersection
    predicted_set = set(predicted_genes)
    ground_truth_set = set(ground_truth_genes)
    
    # Remove essential genes if provided
    if essential_genes:
        essential_set = set(essential_genes)
        predicted_set = predicted_set - essential_set
        ground_truth_set = ground_truth_set - essential_set
    
    # Calculate hits
    hits = predicted_set.intersection(ground_truth_set)
    
    # Calculate hit rate based on predicted genes
    if len(predicted_set) == 0:
        return 0.0
    
    return len(hits) / len(predicted_set)


def calculate_precision_recall(predicted_genes: List[str],
                             ground_truth_genes: List[str],
                             essential_genes: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate precision and recall for predicted genes.
    
    Args:
        predicted_genes: List of predicted gene names
        ground_truth_genes: List of ground truth gene names
        essential_genes: Optional list of essential genes to exclude
        
    Returns:
        Dictionary with precision and recall values
    """
    # Convert to sets
    predicted_set = set(predicted_genes)
    ground_truth_set = set(ground_truth_genes)
    
    # Remove essential genes if provided
    if essential_genes:
        essential_set = set(essential_genes)
        predicted_set = predicted_set - essential_set
        ground_truth_set = ground_truth_set - essential_set
    
    # Calculate hits
    hits = predicted_set.intersection(ground_truth_set)
    
    # Calculate metrics
    precision = len(hits) / len(predicted_set) if len(predicted_set) > 0 else 0.0
    recall = len(hits) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    }


def evaluate_round(predicted_genes: List[str],
                  ground_truth_genes: List[str],
                  round_num: int,
                  essential_genes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Evaluate a single round of predictions.
    
    Args:
        predicted_genes: List of predicted gene names
        ground_truth_genes: List of ground truth gene names
        round_num: Round number
        essential_genes: Optional list of essential genes to exclude
        
    Returns:
        Dictionary with evaluation metrics
    """
    hit_rate = calculate_hit_rate(predicted_genes, ground_truth_genes, essential_genes)
    precision_recall = calculate_precision_recall(predicted_genes, ground_truth_genes, essential_genes)
    
    return {
        "round": round_num,
        "num_predicted": len(predicted_genes),
        "hit_rate": hit_rate,
        **precision_recall,
        "predicted_genes": predicted_genes
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple rounds.
    
    Args:
        results: List of round results
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not results:
        return {}
    
    hit_rates = [r["hit_rate"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1_scores = [r["f1_score"] for r in results]
    
    return {
        "num_rounds": len(results),
        "mean_hit_rate": np.mean(hit_rates),
        "std_hit_rate": np.std(hit_rates),
        "mean_precision": np.mean(precisions),
        "std_precision": np.std(precisions),
        "mean_recall": np.mean(recalls),
        "std_recall": np.std(recalls),
        "mean_f1_score": np.mean(f1_scores),
        "std_f1_score": np.std(f1_scores),
        "hit_rates_by_round": hit_rates,
        "total_unique_genes": len(set(sum([r["predicted_genes"] for r in results], [])))
    }


class BioEvaluator:
    """Main evaluator class for biological discovery."""
    
    def __init__(self, dataset_name: str, essential_genes_path: str = "CEGv2.txt"):
        """Initialize evaluator with dataset."""
        self.dataset_name = dataset_name
        self.essential_genes = self._load_essential_genes(essential_genes_path)
        self.ground_truth = self._load_ground_truth()
    
    def _load_essential_genes(self, path: str) -> List[str]:
        """Load essential genes from file."""
        if os.path.exists(path):
            df = pd.read_csv(path, delimiter='\t')
            return df['GENE'].tolist()
        return []
    
    def _load_ground_truth(self) -> List[str]:
        """Load ground truth genes for dataset."""
        topmovers_path = f"datasets/topmovers_{self.dataset_name}.npy"
        if os.path.exists(topmovers_path):
            topmovers = np.load(topmovers_path, allow_pickle=True)
            
            # Handle Horlbeck dataset special case
            if self.dataset_name == "Horlbeck":
                temp = []
                for pair in topmovers:
                    newpair = f"{pair[0]}_{pair[1]}"
                    temp.append(newpair)
                return temp
            else:
                return topmovers.tolist()
        return []
    
    def evaluate(self, predicted_genes: List[str], round_num: int = 0) -> Dict[str, Any]:
        """Evaluate predicted genes."""
        return evaluate_round(
            predicted_genes,
            self.ground_truth,
            round_num,
            self.essential_genes
        )
    
    def evaluate_multiple_rounds(self, predictions_by_round: List[List[str]]) -> Dict[str, Any]:
        """Evaluate multiple rounds of predictions."""
        results = []
        for i, predictions in enumerate(predictions_by_round):
            results.append(self.evaluate(predictions, i + 1))
        
        return {
            "round_results": results,
            "aggregate": aggregate_results(results)
        }
    
    def get_hits(self, predicted_genes: List[str]) -> List[str]:
        """Get the hit genes from a list of predictions."""
        predicted_set = set(predicted_genes)
        ground_truth_set = set(self.ground_truth)
        
        # Remove essential genes
        if self.essential_genes:
            essential_set = set(self.essential_genes)
            predicted_set = predicted_set - essential_set
            ground_truth_set = ground_truth_set - essential_set
        
        # Return hits
        hits = predicted_set.intersection(ground_truth_set)
        return list(hits)