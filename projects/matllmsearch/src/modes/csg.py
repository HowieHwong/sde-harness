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
from ..utils.materials_oracle import MaterialsOracle
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
            data_path=self.args.data_path,
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
        initial_structures = seed_structures[:self.args.population_size * self.args.parent_size] if seed_structures else []
        # Evaluate initial population and store evaluations
        if initial_structures:
            print(f"Evaluating initial population of {len(initial_structures)} structures...")
            current_evaluations = self.oracle.evaluate(initial_structures)
        else:
            current_evaluations = []
        
        all_generations = []
        all_metrics = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        
        for iteration in range(self.args.max_iter):
            print(f"=== Iteration {iteration + 1}/{self.args.max_iter} ===")
            
            # Generate offspring from current population structures
            current_population_structures = [e.structure for e in current_evaluations] if current_evaluations else []
            if current_population_structures:
                print(f"Generating {self.args.population_size} * {self.args.reproduction_size} offspring from {len(current_population_structures)} parents")
                new_structures = self.structure_generator.generate(
                    current_population_structures, 
                    num_offspring=self.args.reproduction_size
                )
            else:
                print("Zero-shot generation - no parent structures available")
                new_structures = self.structure_generator.generate(
                    [],
                    num_offspring=self.args.population_size
                )
            
            print(f"Generated {len(new_structures)} new structures")
            
            # Track token usage for this iteration
            iteration_input_tokens = 0
            iteration_output_tokens = 0
            iteration_total_tokens = 0
            
            if hasattr(self.structure_generator, '_last_token_usage'):
                for usage in self.structure_generator._last_token_usage:
                    iteration_input_tokens += usage.get("input_tokens", 0)
                    iteration_output_tokens += usage.get("output_tokens", 0)
                    iteration_total_tokens += usage.get("total_tokens", 0)
                
                total_input_tokens += iteration_input_tokens
                total_output_tokens += iteration_output_tokens
                total_tokens += iteration_total_tokens
                
                print(f"Token usage (iteration {iteration + 1}): {iteration_input_tokens} input + {iteration_output_tokens} output = {iteration_total_tokens} total")
            
            if not new_structures:
                print("No valid structures generated, ending optimization")
                break
            
            # Evaluate new structures (children)
            print("Evaluating new structures...")
            child_evaluations = self.oracle.evaluate(new_structures)
            
            # Merge parent evaluations and child evaluations
            all_evaluations = current_evaluations + child_evaluations
            print(f"Merged {len(current_evaluations)} parent evaluations with {len(child_evaluations)} child evaluations")
            
            # Get token usage per structure (distribute iteration tokens across structures)
            num_structures = len(new_structures)
            tokens_per_structure = {
                "input": iteration_input_tokens // num_structures if num_structures > 0 else 0,
                "output": iteration_output_tokens // num_structures if num_structures > 0 else 0,
                "total": iteration_total_tokens // num_structures if num_structures > 0 else 0
            }
            
            # Save generation data (only for new children)
            generation_data = []
            for i, (structure, evaluation) in enumerate(zip(new_structures, child_evaluations)):
                if structure and evaluation:
                    generation_data.append({
                        'Iteration': iteration + 1,
                        'Structure': json.dumps(structure.as_dict()),
                        'Composition': str(structure.composition),
                        'EHullDistance': evaluation.e_hull_distance,
                        'EnergyRelaxed': evaluation.energy_relaxed,
                        'BulkModulusRelaxed': evaluation.bulk_modulus_relaxed,
                        'Objective': evaluation.objective,
                        'Valid': evaluation.valid,
                        'InputTokens': tokens_per_structure["input"],
                        'OutputTokens': tokens_per_structure["output"],
                        'TotalTokens': tokens_per_structure["total"]
                    })
            
            all_generations.extend(generation_data)
            
            # Calculate metrics
            metrics = self.oracle.get_metrics(all_evaluations)
            metrics['iteration'] = iteration + 1
            metrics['input_tokens'] = iteration_input_tokens
            metrics['output_tokens'] = iteration_output_tokens
            metrics['total_tokens'] = iteration_total_tokens
            all_metrics.append(metrics)
            
            print(f"Metrics: {metrics}")
            
            # Update population (select best structures from parents + children)
            valid_evaluations = [e for e in all_evaluations if e.valid]
            if valid_evaluations:
                ranked_evaluations = self.oracle.rank_structures(valid_evaluations, ascending=True)
                # Keep top population_size * parent_size (100 * 2 = 200)
                current_evaluations = ranked_evaluations[:self.args.population_size * self.args.parent_size]
                print(f"Updated population: {len(current_evaluations)} structures (top {self.args.population_size * self.args.parent_size} from {len(valid_evaluations)} valid)")
            else:
                print("No valid structures found, keeping previous population")
            
            # Incremental save after each iteration
            generations_backup_file = output_path / "generations_backup.csv"
            metrics_backup_file = output_path / "metrics_backup.csv"
            
            if generation_data:
                generation_df = pd.DataFrame(generation_data)
                generation_df.to_csv(generations_backup_file, mode='a', header=not generations_backup_file.exists(), index=False)
                print(f"Incremental backup: Appended {len(generation_data)} structures to generations_backup.csv (total: {len(all_generations)})")
            
            if metrics:
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(metrics_backup_file, mode='a', header=not metrics_backup_file.exists(), index=False)
                print(f"Incremental backup: Appended iteration {iteration + 1} metrics to metrics_backup.csv")
        
        # Final save results
        if all_generations:
            generations_df = pd.DataFrame(all_generations)
            generations_df.to_csv(output_path / "generations.csv", index=False)
        
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(output_path / "metrics.csv", index=False)
        
        # Print total token usage summary
        print(f"\n{'='*60}")
        print("Token Usage Summary")
        print(f"{'='*60}")
        print(f"Total Input Tokens: {total_input_tokens:,}")
        print(f"Total Output Tokens: {total_output_tokens:,}")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"{'='*60}\n")
        
        # Save token usage summary to a separate file
        token_summary = {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "iterations": len(all_metrics),
            "average_tokens_per_iteration": total_tokens / len(all_metrics) if all_metrics else 0
        }
        
        import json
        with open(output_path / "token_usage_summary.json", "w") as f:
            json.dump(token_summary, f, indent=2)
        
        results = {
            'total_structures': len(all_generations),
            'iterations': len(all_metrics),
            'final_metrics': all_metrics[-1] if all_metrics else {},
            'output_path': str(output_path),
            'token_usage': token_summary
        }
        
        return results


def run_csg(args) -> Dict[str, Any]:
    """Run Crystal Structure Generation mode"""
    
    # Create and run MatLLMSearch CSG project
    project = MatLLMSearchCSG(args=args)
    results = project.run()
    
    return results