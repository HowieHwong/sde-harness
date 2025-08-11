"""Materials Oracle for evaluating crystal structures using SDE-harness framework"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pymatgen.core.structure import Structure
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


class MaterialsOracle:
    """Oracle for evaluating crystal structures in materials discovery"""
    
    def __init__(self, opt_goal: str = "e_hull_distance", mlip: str = "chgnet", 
                 ppd_path: str = "data/2023-02-07-ppd-mp.pkl.gz", device: str = "cuda"):
        
        self.opt_goal = opt_goal
        self.mlip = mlip
        
        # Initialize stability calculator
        self.stability_calculator = StabilityCalculator(
            mlip=mlip,
            ppd_path=ppd_path,
            device=device
        )
    
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
                stable_count = sum(1 for e in e_hull_distances if e <= 0.03)
                metrics.update({
                    'min_e_hull_distance': min(e_hull_distances),
                    'avg_e_hull_distance': np.mean(e_hull_distances),
                    'stable_structures': stable_count,
                    'stability_rate': stable_count / len(e_hull_distances)
                })
            
            if bulk_moduli:
                metrics.update({
                    'max_bulk_modulus': max(bulk_moduli),
                    'avg_bulk_modulus': np.mean(bulk_moduli)
                })
        
        return metrics