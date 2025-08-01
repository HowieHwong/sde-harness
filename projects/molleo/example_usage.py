#!/usr/bin/env python3
"""
Example usage of the restructured MolLEO with SDE harness
"""

import sys
import os
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.modes import run_single_objective, run_multi_objective


def example_single_objective():
    """Example of single objective optimization"""
    print("=" * 60)
    print("Single Objective Optimization Example")
    print("=" * 60)
    
    # Create args object
    class Args:
        oracle = "qed"  # Optimize for drug-likeness
        model = "openai/gpt-4o-2024-08-06"  # Use GPT-4 for guided mutations
        population_size = 10
        offspring_size = 10
        generations = 3
        mutation_rate = 0.1
        initial_size = 10
        seed = 42
        output_dir = "results/example"
    
    args = Args()
    
    # Run optimization
    results = run_single_objective(args)
    
    print(f"\nBest molecule found: {results['best_molecule']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Total oracle calls: {results['oracle_calls']}")


def example_multi_objective():
    """Example of multi-objective optimization"""
    print("\n" + "=" * 60)
    print("Multi-Objective Optimization Example")
    print("=" * 60)
    
    # Create args object
    class Args:
        max_objectives = ["qed", "logp"]  # Maximize drug-likeness and lipophilicity
        min_objectives = ["sa"]  # Minimize synthetic accessibility, strange request but yea
        mode = "weighted_sum"
        model = "openai/gpt-4o-2024-08-06"  # Use GPT-4 for guided mutations
        population_size = 10
        offspring_size = 10
        generations = 5
        mutation_rate = 0.05
        initial_size = 10
        seed = 42
        weights = [1.0, 0.5, 0.5]  # Weights for [qed, logp, sa]
    
    args = Args()
    
    # Run optimization
    results = run_multi_objective(args)
    
    print(f"\nBest molecule found: {results['best_molecule']}")
    print(f"Best weighted score: {results['best_score']:.4f}")


def example_with_sde_harness_directly():
    """Example using SDE harness components directly"""
    print("\n" + "=" * 60)
    print("Direct SDE Harness Usage Example")
    print("=" * 60)
    
    from sde_harness.core import Generation, Workflow, Oracle, Prompt
    from src.core import MolLEOOptimizer
    from src.oracles import TDCOracle
    
    # Create oracle
    oracle = TDCOracle("sa")
    
    # Create optimizer
    optimizer = MolLEOOptimizer(
        oracle=oracle,
        population_size=10,
        offspring_size=10,
        mutation_rate=0.05,
        model_name="openai/gpt-4o-2024-08-06",
        use_llm_mutations=True
    )
    
    # Initial molecules (example SMILES)
    initial_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(=O)OC1=CC=CC=C1C(=O)O",        # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # Caffeine
    ]
    
    # Run optimization
    results = optimizer.optimize(
        starting_smiles=initial_smiles,
        num_generations=2
    )
    
    print(f"\nBest synthetic accessible molecule found: {results['best_molecule']}")
    print(f"Best SA score: {results['best_score']:.4f}")
    
    # Create a Generation instance for the workflow
    from src.generation import MolLEOGeneration
    generation = MolLEOGeneration(
        model_name="openai/gpt-4o-2024-08-06"
    )
    
    # Create a workflow for multi-round optimization
    workflow = Workflow(
        generator=generation,
        oracle=oracle,
        max_iterations=2,
        enable_history_in_prompts=True
    )
    
    # Define a custom prompt for generating analogs
    from src.core.prompts import MolecularPrompts
    
    # Create prompt function that generates analogs based on history
    def analog_generation_prompt(iteration: int, history: Dict) -> Prompt:
        """Generate prompt for creating molecular analogs"""
        if iteration == 1:
            # First iteration - generate analogs of the best molecule
            prompt_text = MolecularPrompts.ANALOG_GENERATION_PROMPT.format(
                molecule=results['best_molecule'],
                num_analogs=5,
                objective="lower synthetic accessibility score (SA)"
            )
        else:
            # Subsequent iterations - improve based on previous results
            previous_output = history["outputs"][-1]
            previous_scores = history["scores"][-1]
            
            prompt_text = f"""Previous attempt generated these molecules:
{previous_output}

The synthetic accessibility scores were not optimal. Please generate 5 NEW structural analogs of {results['best_molecule']} 
that are likely to have even better (lower) synthetic accessibility scores. Focus on:
- Simpler ring systems
- Common functional groups
- Avoid complex stereochemistry
- Use readily available building blocks

Output ONLY the SMILES strings, one per line."""
        
        return Prompt(custom_template=prompt_text)
    
    # Define custom SA evaluation metric
    def sa_evaluation(prediction: str, reference: str, **kwargs) -> float:
        """Evaluate SA scores of generated molecules"""
        # Extract SMILES from prediction
        lines = prediction.strip().split('\n')
        valid_molecules = []
        total_sa_score = 0
        count = 0
        
        from rdkit import Chem
        for line in lines:
            line = line.strip()
            if line:
                # Try to parse as SMILES
                mol = Chem.MolFromSmiles(line)
                if mol:
                    try:
                        sa_score = oracle.evaluate_molecule(line)
                        valid_molecules.append((line, sa_score))
                        total_sa_score += sa_score
                        count += 1
                    except:
                        pass
        
        # Store results in kwargs for access
        kwargs['generated_molecules'] = valid_molecules
        
        # Return average SA score (lower is better, so we invert for optimization)
        if count > 0:
            avg_sa = total_sa_score / count
            # Normalize to 0-1 where 1 is best (lowest SA)
            return max(0, 1 - (avg_sa / 10))  # Assuming SA scores typically 1-10
        return 0.0
    
    # Register the evaluation metric
    oracle.register_metric("sa_quality", sa_evaluation)
    
    # Run workflow
    print("\nRunning analog generation workflow...")
    workflow_results = workflow.run_sync(
        prompt=analog_generation_prompt,
        reference=f"Generate analogs of {results['best_molecule']}",
        gen_args={
            "model_name": "openai/gpt-4o-2024-08-06",
            "max_tokens": 500,
            "temperature": 0.8,
        },
        metrics=["sa_quality"]
    )
    
    # Extract and display results
    print("\nWorkflow Results:")
    history = workflow_results.get("history", {})
    outputs = history.get("outputs", [])
    
    if outputs:
        print(f"Generated {len(outputs)} iterations")
        
        # Get the last output and parse molecules
        last_output = outputs[-1]
        print(f"\nFinal generated molecules:")
        
        from rdkit import Chem
        lines = last_output.strip().split('\n')
        generated_analogs = []
        
        for line in lines:
            line = line.strip()
            if line:
                mol = Chem.MolFromSmiles(line)
                if mol:
                    try:
                        sa_score = oracle.evaluate_molecule(line)
                        generated_analogs.append((line, sa_score))
                        print(f"  {line}: SA score = {sa_score:.4f}")
                    except:
                        pass
        
        if generated_analogs:
            # Find best analog
            best_analog = min(generated_analogs, key=lambda x: x[1])
            print(f"\nBest analog found: {best_analog[0]} with SA score = {best_analog[1]:.4f}")
            print(f"Original molecule SA score: {results['best_score']:.4f}")


if __name__ == "__main__":
    # Run examples
    # example_single_objective()
    # example_multi_objective()
    example_with_sde_harness_directly()