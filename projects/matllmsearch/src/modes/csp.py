"""Crystal Structure Prediction (CSP) mode implementation"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add sde_harness to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# from sde_harness.core import Generation, Oracle, Workflow
# from sde_harness.base import ProjectBase
from ..evaluators import MaterialsOracle
from ..utils.data_loader import load_seed_structures, matches_composition, matches_unit_cell_pattern
from ..utils.structure_generator import StructureGenerator
from pymatgen.core.composition import Composition


class MatLLMSearchCSP:
    """MatLLMSearch Crystal Structure Prediction project using SDE-harness framework"""
    
    def __init__(self, args):
        """Initialize MatLLMSearch CSP"""
        self.args = args
        
        # Initialize structure generator for CSP
        self.structure_generator = StructureGenerator(
            model=self.args.model,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens,
            fmt=self.args.fmt,
            task="csp",
            args=self.args
        )
        
        # Initialize materials oracle
        self.oracle = MaterialsOracle(
            opt_goal="e_hull_distance",  # CSP focuses on finding ground state
            mlip="orb-v3"  # Use more accurate potential for structure prediction
        )
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the CSP workflow"""
        print(f"Starting Crystal Structure Prediction for {self.args.compound}")
        print(f"Model: {self.args.model}")
        print(f"Population size: {self.args.population_size}, Max iterations: {self.args.max_iter}")
        
        # Load and filter seed structures for target compound
        seed_structures = self._load_target_compound_structures()
        
        # Set random seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        
        # Run CSP workflow (similar to CSG but focused on target compound)
        results = self._run_csp_workflow(seed_structures)
        
        return results
    
    def _run_csp_workflow(self, seed_structures):
        """Run CSP workflow - similar to CSG but focused on target compound"""
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
            
            # Filter for target compound composition
            target_comp = Composition(self.args.compound)
            filtered_structures = []
            for structure in new_structures:
                if structure and (matches_composition(structure.composition, target_comp) or 
                                matches_unit_cell_pattern(structure.composition, target_comp)):
                    filtered_structures.append(structure)
            
            print(f"Generated {len(new_structures)} structures, {len(filtered_structures)} match target compound")
            
            if not filtered_structures:
                print("No structures match target compound, ending optimization")
                break
            
            # Evaluate structures
            print("Evaluating structures...")
            evaluations = self.oracle.evaluate(filtered_structures)
            
            # Save generation data
            generation_data = []
            for i, (structure, evaluation) in enumerate(zip(filtered_structures, evaluations)):
                if structure and evaluation:
                    generation_data.append({
                        'Iteration': iteration + 1,
                        'Structure': json.dumps(structure.as_dict()),
                        'Composition': str(structure.composition),
                        'EHullDistance': evaluation.e_hull_distance,
                        'EnergyRelaxed': evaluation.energy_relaxed,
                        'Objective': evaluation.objective,
                        'Valid': evaluation.valid
                    })
            
            all_generations.extend(generation_data)
            
            # Calculate metrics
            metrics = self.oracle.get_metrics(evaluations)
            metrics['iteration'] = iteration + 1
            all_metrics.append(metrics)
            
            print(f"Metrics: {metrics}")
            
            # Update population (select best structures for CSP)
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
            'output_path': str(output_path),
            'target_compound': self.args.compound
        }
        
        return results
    
    def _load_target_compound_structures(self):
        """Load seed structures that match the target compound pattern"""
        
        # Load all available seed structures
        all_seeds = load_seed_structures(
            data_path=self.args.data_path,  # Use specified data path
            task="csp",
            random_seed=self.args.seed
        )
        
        if not all_seeds:
            print(f"No seed structures found, using zero-shot generation for {self.args.compound}")
            return []
        
        # Filter structures that match the target compound pattern
        target_comp = Composition(self.args.compound)
        matching_structures = []
        
        for structure in all_seeds:
            if structure is None:
                continue
                
            try:
                # Check if structure has the same unit cell pattern as target
                if matches_unit_cell_pattern(structure.composition, target_comp):
                    matching_structures.append(structure)
                # Also include exact composition matches
                elif matches_composition(structure.composition, target_comp):
                    matching_structures.append(structure)
            except Exception as e:
                print(f"Error checking structure composition: {e}")
                continue
        
        print(f"Found {len(matching_structures)} seed structures matching {self.args.compound} pattern")
        
        # If no matching structures, use a broader search
        if not matching_structures:
            print(f"No exact matches found, using structures with similar element count")
            target_elem_count = len(target_comp.elements)
            
            for structure in all_seeds[:100]:  # Limit to first 100 for efficiency
                if structure is None:
                    continue
                    
                try:
                    if len(structure.composition.elements) == target_elem_count:
                        matching_structures.append(structure)
                except Exception:
                    continue
            
            print(f"Found {len(matching_structures)} structures with {target_elem_count} elements")
        
        return matching_structures[:50]  # Limit to reasonable number


def run_csp(args) -> Dict[str, Any]:
    """Run Crystal Structure Prediction mode"""
    
    # Validate compound choice
    valid_compounds = ["Ag6O2", "Bi2F8", "Co2Sb2", "Co4B2", "Cr4Si4", "KZnF3", "Sr2O4", "YMg3"]
    if args.compound not in valid_compounds:
        raise ValueError(f"Compound {args.compound} not supported. Choose from: {valid_compounds}")
    
    # Create and run MatLLMSearch CSP project
    project = MatLLMSearchCSP(args=args)
    results = project.run()
    
    return results