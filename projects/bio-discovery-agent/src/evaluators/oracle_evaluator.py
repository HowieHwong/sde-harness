"""Bio evaluator using sde_harness.core.Oracle for metrics."""
import os
import sys
from typing import List, Dict, Any, Set, Tuple
import pandas as pd
import numpy as np

# Add sde-harness to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, project_root)

from sde_harness.core import Oracle


class BioOracleEvaluator:
    """Evaluator for biological discovery using Oracle framework."""
    
    def __init__(self, data_name: str, essential_genes_path: str = "CEGv2.txt"):
        self.data_name = data_name
        self.oracle = Oracle()
        self.essential_genes = self._load_essential_genes(essential_genes_path)
        self._setup_metrics()
        self._load_ground_truth()
        
    def _load_ground_truth(self):
        """Load ground truth data."""
        try:
            # Load hit genes from topmovers file (same as original BioEvaluator)
            topmovers_path = f'datasets/topmovers_{self.data_name}.npy'
            if os.path.exists(topmovers_path):
                topmovers = np.load(topmovers_path, allow_pickle=True)
                
                # Handle Horlbeck dataset special case
                if self.data_name == "Horlbeck":
                    temp = []
                    for pair in topmovers:
                        newpair = f"{pair[0]}_{pair[1]}"
                        temp.append(newpair)
                    self.hit_genes = set(temp)
                else:
                    self.hit_genes = set(topmovers.tolist())
            else:
                print(f"Warning: Could not find topmovers file: {topmovers_path}")
                self.hit_genes = set()
                
            # Load ground truth with scores
            ground_truth_path = f'datasets/ground_truth_{self.data_name}.csv'
            if os.path.exists(ground_truth_path):
                self.ground_truth = pd.read_csv(ground_truth_path, index_col=0)
            else:
                self.ground_truth = pd.DataFrame()
            
        except Exception as e:
            print(f"Warning: Could not load ground truth data: {e}")
            self.hit_genes = set()
            self.ground_truth = pd.DataFrame()
    
    def _load_essential_genes(self, path: str) -> Set[str]:
        """Load essential genes from file."""
        if os.path.exists(path):
            df = pd.read_csv(path, delimiter='\t')
            return set(df['GENE'].tolist())
        return set()
            
    def _setup_metrics(self):
        """Register all bio-specific metrics with Oracle."""
        
        # Single-round metrics
        def hit_rate_metric(prediction: List[str], reference: Set[str], **kwargs) -> float:
            """Calculate hit rate for predicted genes."""
            if not prediction:
                return 0.0
            # Filter out essential genes
            essential_genes = kwargs.get('essential_genes', set())
            filtered_predictions = [g for g in prediction if g not in essential_genes]
            filtered_reference = reference - essential_genes
            
            if not filtered_predictions:
                return 0.0
            hits = sum(1 for gene in filtered_predictions if gene in filtered_reference)
            return hits / len(filtered_predictions)
            
        def precision_at_k_metric(prediction: List[str], reference: Set[str], k: int = 10, **kwargs) -> float:
            """Calculate precision at k."""
            # Filter out essential genes
            essential_genes = kwargs.get('essential_genes', set())
            filtered_predictions = [g for g in prediction if g not in essential_genes]
            filtered_reference = reference - essential_genes
            
            top_k = filtered_predictions[:k] if len(filtered_predictions) >= k else filtered_predictions
            if not top_k:
                return 0.0
            hits = sum(1 for gene in top_k if gene in filtered_reference)
            return hits / len(top_k)
            
        # Register single-round metrics
        self.oracle.register_metric("hit_rate", hit_rate_metric)
        self.oracle.register_metric("precision_at_10", lambda p, r, **kw: precision_at_k_metric(p, r, k=10, **kw))
        self.oracle.register_metric("precision_at_50", lambda p, r, **kw: precision_at_k_metric(p, r, k=50, **kw))
        
        # Multi-round metrics
        def cumulative_hits_metric(history: Dict[str, List[Any]], reference: Set[str], 
                                 current_iteration: int, prediction: List[str] = None, **kwargs) -> float:
            """Track cumulative unique hits across all rounds."""
            essential_genes = kwargs.get('essential_genes', set())
            filtered_reference = reference - essential_genes
            
            all_predictions = []
            for output in history.get("outputs", []):
                if isinstance(output, list):
                    all_predictions.extend(output)
            
            # Add current prediction if provided
            if prediction:
                all_predictions.extend(prediction)
                
            # Filter out essential genes
            filtered_predictions = [g for g in all_predictions if g not in essential_genes]
            unique_hits = set(gene for gene in filtered_predictions if gene in filtered_reference)
            return float(len(unique_hits))
            
        def discovery_efficiency_metric(history: Dict[str, List[Any]], reference: Set[str], 
                                      current_iteration: int, **kwargs) -> float:
            """Calculate efficiency of discovery (hits per gene tested)."""
            essential_genes = kwargs.get('essential_genes', set())
            filtered_reference = reference - essential_genes
            
            all_predictions = []
            for output in history.get("outputs", []):
                if isinstance(output, list):
                    all_predictions.extend(output)
                    
            if not all_predictions:
                return 0.0
                
            # Filter out essential genes
            filtered_predictions = [g for g in all_predictions if g not in essential_genes]
            unique_predictions = set(filtered_predictions)
            unique_hits = set(gene for gene in unique_predictions if gene in filtered_reference)
            
            if not unique_predictions:
                return 0.0
            
            return len(unique_hits) / len(unique_predictions)
            
        def hit_progression_metric(history: Dict[str, List[Any]], reference: Set[str], 
                                 current_iteration: int, **kwargs) -> float:
            """Calculate the rate of new hits discovered per round."""
            hits_per_round = []
            seen_genes = set()
            
            for output in history.get("outputs", []):
                if isinstance(output, list):
                    round_hits = 0
                    for gene in output:
                        if gene not in seen_genes and gene in reference:
                            round_hits += 1
                        seen_genes.add(gene)
                    hits_per_round.append(round_hits)
                    
            if len(hits_per_round) < 2:
                return 0.0
                
            # Calculate average improvement rate
            return np.mean(hits_per_round)
            
        # Register multi-round metrics
        self.oracle.register_multi_round_metric("cumulative_hits", cumulative_hits_metric)
        self.oracle.register_multi_round_metric("discovery_efficiency", discovery_efficiency_metric)
        self.oracle.register_multi_round_metric("hit_progression", hit_progression_metric)
        
    def evaluate(self, predicted_genes: List[str], round_num: int = None) -> Dict[str, Any]:
        """Evaluate a single round of predictions."""
        # Basic validation
        valid_genes = [g for g in predicted_genes if g in self.ground_truth.index] if not self.ground_truth.empty else predicted_genes
        
        # Compute single-round metrics with essential genes
        results = self.oracle.compute(
            prediction=valid_genes,
            reference=self.hit_genes,
            metrics=["hit_rate", "precision_at_10", "precision_at_50"],
            essential_genes=self.essential_genes
        )
        
        # Add additional info
        # Count hits after filtering essential genes
        filtered_valid_genes = [g for g in valid_genes if g not in self.essential_genes]
        filtered_hits = self.hit_genes - self.essential_genes
        hit_genes = [g for g in filtered_valid_genes if g in filtered_hits]
        
        results.update({
            "round": round_num,
            "predicted_genes": predicted_genes,
            "valid_genes": valid_genes,
            "num_hits": len(hit_genes),
            "hit_genes": hit_genes
        })
        
        return results
        
    def evaluate_with_history(self, current_predictions: List[str], 
                            history: Dict[str, List[Any]], 
                            iteration: int) -> Dict[str, Any]:
        """Evaluate current round with historical context."""
        # Get single-round metrics
        single_round_results = self.evaluate(current_predictions, iteration)
        
        # Compute multi-round metrics
        multi_round_results = self.oracle.compute_with_history(
            prediction=current_predictions,
            reference=self.hit_genes,
            history=history,
            current_iteration=iteration,
            metrics=["cumulative_hits", "discovery_efficiency", "hit_progression"],
            essential_genes=self.essential_genes
        )
        
        # Combine results
        single_round_results.update(multi_round_results)
        
        # Add trend analysis
        if history.get("scores"):
            trend_metrics = self.oracle.compute_trend_metrics(history, "hit_rate")
            single_round_results["trends"] = trend_metrics
            
        return single_round_results
        
    def get_hits(self, predicted_genes: List[str]) -> List[str]:
        """Get list of hit genes from predictions."""
        # Filter out essential genes from both predicted and hit genes
        filtered_predictions = [g for g in predicted_genes if g not in self.essential_genes]
        filtered_hits = self.hit_genes - self.essential_genes
        return [g for g in filtered_predictions if g in filtered_hits]
        
    def get_gene_scores(self, genes: List[str]) -> List[Tuple[str, float]]:
        """Get scores for a list of genes."""
        gene_scores = []
        for gene in genes:
            if gene in self.ground_truth.index:
                score = self.ground_truth.loc[gene].values[0] if hasattr(self.ground_truth.loc[gene], 'values') else self.ground_truth.loc[gene]
                gene_scores.append((gene, float(score)))
        return gene_scores
        
    def evaluate_multiple_rounds(self, all_predictions: List[List[str]]) -> Dict[str, Any]:
        """Evaluate multiple rounds of predictions."""
        history = {
            "prompts": [],
            "outputs": [],
            "scores": []
        }
        
        results_per_round = []
        
        for i, predictions in enumerate(all_predictions):
            # Evaluate with history
            round_results = self.evaluate_with_history(
                predictions, history, i + 1
            )
            results_per_round.append(round_results)
            
            # Update history
            history["outputs"].append(predictions)
            history["scores"].append({
                "hit_rate": round_results["hit_rate"],
                "precision_at_10": round_results.get("precision_at_10", 0.0)
            })
            
        # Aggregate results
        all_predicted = []
        all_hits = []
        
        for result in results_per_round:
            all_predicted.extend(result["valid_genes"])
            all_hits.extend(result["hit_genes"])
            
        unique_predicted = list(set(all_predicted))
        unique_hits = list(set(all_hits))
        
        final_results = {
            "rounds": results_per_round,
            "aggregate": {
                "total_unique_genes": len(unique_predicted),
                "total_unique_hits": len(unique_hits),
                "mean_hit_rate": np.mean([r["hit_rate"] for r in results_per_round]),
                "final_efficiency": len(unique_hits) / len(unique_predicted) if unique_predicted else 0.0,
                "unique_predicted_genes": unique_predicted,
                "unique_hit_genes": unique_hits
            }
        }
        
        # Add final multi-round metrics
        if results_per_round:
            final_multi_metrics = results_per_round[-1]
            final_results["final_metrics"] = {
                "cumulative_hits": final_multi_metrics.get("cumulative_hits", 0),
                "discovery_efficiency": final_multi_metrics.get("discovery_efficiency", 0.0),
                "hit_progression": final_multi_metrics.get("hit_progression", 0.0)
            }
            
        return final_results