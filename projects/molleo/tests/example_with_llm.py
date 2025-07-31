#!/usr/bin/env python3
"""
Example usage of MolLEO with LLM calls
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Add the molleo project root to path
molleo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, molleo_root)

from src.core import MolLEOOptimizer
from src.oracles import TDCOracle
from src.modes import run_single_objective


def test_llm_mutation():
    """Test LLM-guided molecular mutation"""
    print("=" * 60)
    print("Testing LLM-Guided Molecular Mutation")
    print("=" * 60)
    
    # Create oracle
    oracle = TDCOracle("qed")
    
    # Create optimizer with LLM
    print("\nInitializing optimizer with LLM support...")
    optimizer = MolLEOOptimizer(
        oracle=oracle,
        population_size=10,
        offspring_size=20,
        mutation_rate=0.01,
        model_name="openai/gpt-4o-mini",  # Using smaller model for testing
        use_llm_mutations=True
    )
    
    # Test with a single molecule
    test_molecule = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    print(f"\nTest molecule: {test_molecule}")
    print(f"Initial QED: {oracle.evaluate_molecule(test_molecule):.4f}")
    
    # Initialize small population
    print("\nRunning 1 generation with LLM mutations...")
    results = optimizer.optimize(
        starting_smiles=[test_molecule],
        num_generations=1
    )
    
    print(f"\nResults after 1 generation:")
    print(f"Best molecule: {results['best_molecule']}")
    print(f"Best QED: {results['best_score']:.4f}")
    print(f"Population size: {len(results['final_population'])}")
    
    # Show some generated molecules
    print("\nSome molecules from final population:")
    for i, (smiles, score) in enumerate(results['final_population'][:5]):
        print(f"{i+1}. {smiles}")
        print(f"   QED: {score:.4f}")


def test_single_objective_mode():
    """Test single objective optimization with LLM"""
    print("\n" + "=" * 60)
    print("Testing Single Objective Mode with LLM")
    print("=" * 60)
    
    class Args:
        oracle = "qed"
        mol_lm = "GPT-4"
        model = "openai/gpt-4o-mini"  # Using mini model for cost
        population_size = 20
        offspring_size = 40
        generations = 2  # Just 2 generations for testing
        mutation_rate = 0.01
        initial_size = 5
        seed = 42
        output_dir = "results/llm_test"
        n_jobs = -1
    
    args = Args()
    
    print(f"\nRunning optimization:")
    print(f"- Oracle: {args.oracle}")
    print(f"- Model: {args.model}")
    print(f"- Generations: {args.generations}")
    print(f"- Population: {args.population_size}")
    
    results = run_single_objective(args)
    
    return results


def test_llm_prompt_only():
    """Test just the LLM prompt generation without full optimization"""
    print("\n" + "=" * 60)
    print("Testing LLM Prompt Generation")
    print("=" * 60)
    
    from src.core.prompts import MolecularPrompts
    from sde_harness.core import Generation
    
    # Initialize generator
    generator = Generation(
        models_file=os.path.join(project_root, "models.yaml"),
        credentials_file=os.path.join(project_root, "credentials.yaml")
    )
    
    # Create a mutation prompt
    parent_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    prompt = MolecularPrompts.get_mutation_prompt(
        parent_smiles=parent_smiles,
        num_mutations=3
    )
    
    print(f"\nParent molecule: {parent_smiles} (Aspirin)")
    print("\nPrompt preview:")
    print("-" * 40)
    print(prompt.build()[:300] + "...")
    print("-" * 40)
    
    # Generate mutations
    print("\nGenerating mutations with LLM...")
    try:
        response = generator.generate(
            prompt=prompt.build(),
            model_name="openai/gpt-4o-mini",
            temperature=0.8,
            max_tokens=200
        )
        
        print("\nLLM Response:")
        print(response['text'])
        
        # Try to parse SMILES
        lines = response['text'].strip().split('\n')
        valid_mutations = []
        
        from src.utils import smiles_to_mol
        for line in lines:
            line = line.strip()
            # Remove leading numbers and dots (e.g., "1. SMILES" -> "SMILES")
            if line and line[0].isdigit():
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    line = parts[1].strip()
                else:
                    continue
            
            if line and not line.startswith('#'):
                mol = smiles_to_mol(line)
                if mol:
                    valid_mutations.append(line)
        
        print(f"\nValid mutations found: {len(valid_mutations)}")
        for i, smi in enumerate(valid_mutations, 1):
            print(f"{i}. {smi}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your API credentials are configured correctly.")
    
    generator.close()


def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description="Test MolLEO with LLM")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--full", action="store_true", help="Run full optimization")
    args = parser.parse_args()
    
    if args.quick:
        # Just test prompt generation
        test_llm_prompt_only()
    elif args.full:
        # Run full optimization
        test_single_objective_mode()
    else:
        # Run all tests
        test_llm_prompt_only()
        test_llm_mutation()
        test_single_objective_mode()


if __name__ == "__main__":
    main()