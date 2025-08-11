"""Crystal Structure Generation (CSG) mode implementation"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add sde_harness to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# from sde_harness.core import Generation, Oracle, Workflow
from sde_harness.base import ProjectBase
from ..evaluators import MaterialsOracle
from ..utils.data_loader import load_seed_structures
from ..utils.structure_generator import StructureGenerator


class MatLLMSearchCSG:
    """MatLLMSearch Crystal Structure Generation project"""
    
    def __init__(self, args):
        """Initialize MatLLMSearch CSG"""
        self.args = args
        
        # Initialize structure generator
        self.structure_generator = StructureGenerator(
            model=self.args.model,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens,
            tensor_parallel_size=self.args.tensor_parallel_size,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            fmt=self.args.fmt,
            task="csg",
            args=self.args
        )
        
        # Initialize materials oracle
        self.oracle = MaterialsOracle(
            opt_goal=self.args.opt_goal,
            mlip="chgnet" if self.args.opt_goal == "e_hull_distance" else "orb-v3"
        )
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the CSG workflow"""
        print(f"Starting Crystal Structure Generation with {self.args.model}")
        print(f"Population size: {self.args.population_size}, Max iterations: {self.args.max_iter}")
        
        # Load seed structures
        seed_structures = load_seed_structures(
            pool_size=self.args.pool_size,
            task="csg",
            random_seed=self.args.seed
        )
        
        # Set random seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        
        # Run custom workflow
        results = self._run_evolutionary_workflow(seed_structures)
        
        return results
    
    def _run_evolutionary_workflow(self, seed_structures):
        """Run the evolutionary optimization workflow"""
        from pathlib import Path
        import pandas as pd
        import json
        
        # Create output directory
        output_path = Path(self.args.log_dir) / self.args.save_label
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize population
        current_population = seed_structures[:self.args.population_size] if seed_structures else []
        
        all_generations = []
        all_metrics = []
        
        for iteration in range(self.args.max_iter):
            print(f"\n=== Iteration {iteration + 1}/{self.args.max_iter} ===")
            
            # Generate offspring
            if current_population:
                print(f"Generating {self.args.reproduction_size} offspring from {len(current_population)} parents")
                new_structures = self.structure_generator.generate(
                    current_population, 
                    num_offspring=self.args.reproduction_size
                )
            else:
                print("Zero-shot generation - no parent structures available")
                new_structures = self.structure_generator.generate(
                    [],
                    num_offspring=self.args.population_size
                )
            
            print(f"Generated {len(new_structures)} new structures")
            
            if not new_structures:
                print("No valid structures generated, ending optimization")
                break
            
            # Evaluate structures
            print("Evaluating structures...")
            evaluations = self.oracle.evaluate(new_structures)
            
            # Save generation data
            generation_data = []
            for i, (structure, evaluation) in enumerate(zip(new_structures, evaluations)):
                if structure and evaluation:
                    generation_data.append({
                        'Iteration': iteration + 1,
                        'Structure': json.dumps(structure.as_dict()),
                        'Composition': str(structure.composition),
                        'EHullDistance': evaluation.e_hull_distance,
                        'EnergyRelaxed': evaluation.energy_relaxed,
                        'BulkModulusRelaxed': evaluation.bulk_modulus_relaxed,
                        'Objective': evaluation.objective,
                        'Valid': evaluation.valid
                    })
            
            all_generations.extend(generation_data)
            
            # Calculate metrics
            metrics = self.oracle.get_metrics(evaluations)
            metrics['iteration'] = iteration + 1
            all_metrics.append(metrics)
            
            print(f"Metrics: {metrics}")
            
            # Update population (select best structures)
            valid_evaluations = [e for e in evaluations if e.valid]
            if valid_evaluations:
                ranked_evaluations = self.oracle.rank_structures(valid_evaluations, ascending=True)
                current_population = [e.structure for e in ranked_evaluations[:self.args.population_size]]
                print(f"Updated population: {len(current_population)} structures")
            else:
                print("No valid structures found, keeping previous population")
        
        # Save results
        if all_generations:
            generations_df = pd.DataFrame(all_generations)
            generations_df.to_csv(output_path / "generations.csv", index=False)
        
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(output_path / "metrics.csv", index=False)
        
        results = {
            'total_structures': len(all_generations),
            'iterations': len(all_metrics),
            'final_metrics': all_metrics[-1] if all_metrics else {},
            'output_path': str(output_path)
        }
        
        return results


def run_csg(args) -> Dict[str, Any]:
    """Run Crystal Structure Generation mode"""
    
    # Create and run MatLLMSearch CSG project
    project = MatLLMSearchCSG(args=args)
    results = project.run()
    
    return results