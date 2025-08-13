"""Materials Oracle for evaluating crystal structures using SDE-harness framework"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pymatgen.core.structure import Structure
from sde_harness.core.oracle import Oracle
from ..utils.stability_calculator import StabilityCalculator, StabilityResult


@dataclass
class MaterialsEvaluation:
    """Evaluation result for a material structure"""
    structure: Structure
    energy: float = np.inf
    energy_relaxed: float = np.inf
    e_hull_distance: float = np.inf
    bulk_modulus: float = -np.inf
    bulk_modulus_relaxed: float = -np.inf
    delta_e: float = np.inf
    objective: float = np.inf
    structure_relaxed: Optional[Structure] = None
    valid: bool = False


class MaterialsOracle(Oracle):
    """Oracle for evaluating crystal structures in materials discovery
    
    Extends SDE-Harness Oracle with materials-specific evaluation metrics.
    """
    
    def __init__(self, opt_goal: str = "e_hull_distance", mlip: str = "chgnet", 
                 ppd_path: str = "data/2023-02-07-ppd-mp.pkl.gz", device: str = "cuda"):
        
        # Initialize parent Oracle class
        super().__init__()
        
        self.opt_goal = opt_goal
        self.mlip = mlip
        
        # Initialize stability calculator
        self.stability_calculator = StabilityCalculator(
            mlip=mlip,
            ppd_path=ppd_path,
            device=device
        )
        
        # Register materials-specific metrics
        self._register_materials_metrics()
    
    def _register_materials_metrics(self) -> None:
        """Register materials-specific metrics with the Oracle base class"""
        
        # Register stability metric
        def stability_metric(prediction: MaterialsEvaluation, reference: Any, **kwargs) -> float:
            """Evaluate stability (E_hull distance) - lower is better"""
            if not prediction.valid:
                return float('inf')
            return prediction.e_hull_distance
        
        # Register bulk modulus metric  
        def bulk_modulus_metric(prediction: MaterialsEvaluation, reference: Any, **kwargs) -> float:
            """Evaluate bulk modulus - higher is better, so return negative for minimization"""
            if not prediction.valid:
                return float('-inf')
            return -prediction.bulk_modulus_relaxed
        
        # Register multi-objective metric
        def multi_objective_metric(prediction: MaterialsEvaluation, reference: Any, **kwargs) -> float:
            """Multi-objective score combining stability and bulk modulus"""
            if not prediction.valid:
                return float('inf')
            return prediction.objective
        
        # Register validity metric
        def validity_metric(prediction: MaterialsEvaluation, reference: Any, **kwargs) -> float:
            """Structure validity (1.0 if valid, 0.0 if invalid)"""
            return float(prediction.valid)
        
        # Register the metrics
        self.register_metric("stability", stability_metric)
        self.register_metric("bulk_modulus", bulk_modulus_metric) 
        self.register_metric("multi_objective", multi_objective_metric)
        self.register_metric("validity", validity_metric)
        
        # Register multi-round metrics for materials optimization
        def materials_improvement_rate(history: Dict[str, List[Any]], reference: Any, 
                                     current_iteration: int, **kwargs) -> float:
            """Calculate improvement rate for materials metrics"""
            target_metric = kwargs.get("target_metric", "stability")
            if not history.get("scores") or len(history["scores"]) < 2:
                return 0.0
            
            scores = []
            for score_dict in history["scores"]:
                if isinstance(score_dict, dict) and target_metric in score_dict:
                    scores.append(score_dict[target_metric])
            
            if len(scores) < 2:
                return 0.0
            
            # For stability, improvement means lower values
            if target_metric == "stability":
                return (scores[0] - scores[-1]) / len(scores)
            else:
                return (scores[-1] - scores[0]) / len(scores)
        
        def convergence_rate(history: Dict[str, List[Any]], reference: Any,
                           current_iteration: int, **kwargs) -> float:
            """Calculate convergence rate for materials optimization"""
            if not history.get("scores") or len(history["scores"]) < 3:
                return 0.0
            
            # Check convergence based on stability scores
            recent_scores = []
            for score_dict in history["scores"][-3:]:
                if isinstance(score_dict, dict) and "stability" in score_dict:
                    recent_scores.append(score_dict["stability"])
            
            if len(recent_scores) < 2:
                return 0.0
            
            # Calculate variance of recent scores (lower variance = more converged)
            mean_score = sum(recent_scores) / len(recent_scores)
            variance = sum((score - mean_score) ** 2 for score in recent_scores) / len(recent_scores)
            
            # Return inverse of variance (higher = more converged)
            return 1.0 / (1.0 + variance)
        
        # Register multi-round metrics
        self.register_multi_round_metric("materials_improvement", materials_improvement_rate)
        self.register_multi_round_metric("convergence_rate", convergence_rate)
    
    def evaluate(self, structures: List[Structure]) -> List[MaterialsEvaluation]:
        """Evaluate a list of structures"""
        if not structures:
            return []
        
        # Compute stability results
        wo_ehull = (self.opt_goal == "bulk_modulus_relaxed")
        wo_bulk = (self.opt_goal == "e_hull_distance")
        
        stability_results = self.stability_calculator.compute_stability(
            structures,
            wo_ehull=wo_ehull,
            wo_bulk=wo_bulk
        )
        
        # Convert to MaterialsEvaluation objects
        evaluations = []
        for i, (structure, stability_result) in enumerate(zip(structures, stability_results)):
            evaluation = self._create_evaluation(structure, stability_result)
            evaluations.append(evaluation)
        
        return evaluations
    
    def _create_evaluation(self, structure: Structure, 
                          stability_result: Optional[StabilityResult]) -> MaterialsEvaluation:
        """Create MaterialsEvaluation from stability result"""
        
        if stability_result is None:
            return MaterialsEvaluation(
                structure=structure,
                valid=False
            )
        
        # Calculate objective based on optimization goal
        if self.opt_goal == "e_hull_distance":
            objective = stability_result.e_hull_distance
        elif self.opt_goal == "bulk_modulus_relaxed":
            objective = -stability_result.bulk_modulus_relaxed  # Negative for maximization
        elif self.opt_goal == "multi-obj":
            objective = self._multi_objective_score(stability_result)
        else:
            objective = stability_result.e_hull_distance
        
        return MaterialsEvaluation(
            structure=structure,
            energy=stability_result.energy,
            energy_relaxed=stability_result.energy_relaxed,
            e_hull_distance=stability_result.e_hull_distance,
            bulk_modulus=stability_result.bulk_modulus,
            bulk_modulus_relaxed=stability_result.bulk_modulus_relaxed,
            delta_e=stability_result.delta_e,
            structure_relaxed=stability_result.structure_relaxed,
            objective=objective,
            valid=self._is_valid_result(stability_result)
        )
    
    def _multi_objective_score(self, result: StabilityResult, 
                              e_weight: float = 0.7, b_weight: float = 0.3) -> float:
        """Calculate multi-objective score combining stability and bulk modulus"""
        
        e_hull = result.e_hull_distance
        bulk_mod = result.bulk_modulus_relaxed
        
        # Handle invalid values
        if np.isnan(e_hull) or np.isinf(e_hull):
            e_hull = 1.0  # High penalty for invalid stability
        
        if np.isnan(bulk_mod) or np.isinf(bulk_mod) or bulk_mod <= 0:
            bulk_mod = 0.0  # No reward for invalid bulk modulus
        
        # Normalize and combine (lower is better for this objective)
        e_score = min(e_hull, 1.0)  # Cap at 1.0 eV/atom
        b_score = 1.0 / (1.0 + bulk_mod / 100.0)  # Inverse relationship for bulk modulus
        
        return e_weight * e_score + b_weight * b_score
    
    def _is_valid_result(self, result: StabilityResult) -> bool:
        """Check if stability result is valid"""
        if result is None:
            return False
        
        # Check for valid energy
        if np.isnan(result.energy) or np.isinf(result.energy):
            return False
        
        # Check for valid stability (if computed)
        if self.opt_goal != "bulk_modulus_relaxed":
            if np.isnan(result.e_hull_distance) or np.isinf(result.e_hull_distance):
                return False
        
        # Check for valid bulk modulus (if computed)
        if self.opt_goal != "e_hull_distance":
            if np.isnan(result.bulk_modulus_relaxed) or result.bulk_modulus_relaxed <= 0:
                return False
        
        return True
    
    def rank_structures(self, evaluations: List[MaterialsEvaluation], 
                       ascending: bool = True) -> List[MaterialsEvaluation]:
        """Rank structures by objective value"""
        valid_evaluations = [eval for eval in evaluations if eval.valid]
        
        if not valid_evaluations:
            return evaluations
        
        # Sort by objective value
        sorted_evaluations = sorted(
            valid_evaluations,
            key=lambda x: x.objective,
            reverse=not ascending
        )
        
        # Add invalid evaluations at the end
        invalid_evaluations = [eval for eval in evaluations if not eval.valid]
        
        return sorted_evaluations + invalid_evaluations
    
    def get_metrics(self, evaluations: List[MaterialsEvaluation]) -> Dict[str, Any]:
        """Calculate summary metrics for evaluations"""
        if not evaluations:
            return {}
        
        valid_evals = [eval for eval in evaluations if eval.valid]
        
        metrics = {
            'total_structures': len(evaluations),
            'valid_structures': len(valid_evals),
            'validity_rate': len(valid_evals) / len(evaluations) if evaluations else 0.0
        }
        
        if valid_evals:
            objectives = [eval.objective for eval in valid_evals]
            e_hull_distances = [eval.e_hull_distance for eval in valid_evals 
                               if not (np.isnan(eval.e_hull_distance) or np.isinf(eval.e_hull_distance))]
            bulk_moduli = [eval.bulk_modulus_relaxed for eval in valid_evals 
                          if not (np.isnan(eval.bulk_modulus_relaxed) or eval.bulk_modulus_relaxed <= 0)]
            
            metrics.update({
                'best_objective': min(objectives),
                'avg_objective': np.mean(objectives),
                'worst_objective': max(objectives),
            })
            
            if e_hull_distances:
                stable_count_003 = sum(1 for e in e_hull_distances if e <= 0.03)
                stable_count_01 = sum(1 for e in e_hull_distances if e <= 0.1)
                metrics.update({
                    'min_e_hull_distance': min(e_hull_distances),
                    'avg_e_hull_distance': np.mean(e_hull_distances),
                    'stable_structures_003': stable_count_003,
                    'stable_structures_01': stable_count_01,
                    'stability_rate_003': stable_count_003 / len(e_hull_distances),
                    'stability_rate_01': stable_count_01 / len(e_hull_distances),
                    'metastability_rate_003': stable_count_003 / len(valid_evals),
                    'metastability_rate_01': stable_count_01 / len(valid_evals)
                })
            
            if bulk_moduli:
                metrics.update({
                    'max_bulk_modulus': max(bulk_moduli),
                    'avg_bulk_modulus': np.mean(bulk_moduli)
                })
        
        return metrics
    
    # Override Oracle methods to integrate with SDE-Harness
    def compute_structures(self, structures: List[Structure], reference: Any = None,
                          metrics: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """Compute metrics for structures using SDE-Harness Oracle interface"""
        
        # Evaluate structures to get MaterialsEvaluation objects
        evaluations = self.evaluate(structures)
        
        # Use specified metrics or default to all registered metrics
        metrics_to_compute = metrics or self.list_single_round_metrics()
        
        # Compute metrics using parent Oracle class
        results = {metric: [] for metric in metrics_to_compute}
        
        for evaluation in evaluations:
            for metric in metrics_to_compute:
                score = super().compute(evaluation, reference, [metric]).get(metric, float('inf'))
                results[metric].append(score)
        
        return results