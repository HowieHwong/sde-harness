#!/usr/bin/env python3
"""
Comprehensive example of MolLEO with SDE Harness
Shows both single and multi-objective optimization with LLMs
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

from src.modes import run_single_objective, run_multi_objective
from src.core import MolLEOOptimizer
from src.oracles import TDCOracle
import weave


def demo_single_objective():
    """Demo single objective optimization"""
    print("\n" + "=" * 80)
    print("DEMO: Single Objective Optimization - QED with LLM")
    print("=" * 80)
    
    weave.init("molleo_demo_single")
    
    class Args:
        oracle = "qed"
        mol_lm = "GPT-4"
        model = "openai/gpt-4o-mini"
        population_size = 20
        offspring_size = 40
        generations = 3
        mutation_rate = 0.01
        initial_size = 5
        seed = 42
        n_jobs = -1
    
    args = Args()
    
    print(f"\nConfiguration:")
    print(f"- Target: {args.oracle} (drug-likeness)")
    print(f"- Model: {args.model}")
    print(f"- Population: {args.population_size}")
    print(f"- Generations: {args.generations}")
    
    results = run_single_objective(args)
    
    print(f"\n✅ Optimization Complete!")
    print(f"   Best QED: {results['best_score']:.4f}")
    print(f"   Best molecule: {results['best_molecule']}")
    
    return results


def demo_multi_objective():
    """Demo multi-objective optimization"""
    print("\n" + "=" * 80)
    print("DEMO: Multi-Objective Optimization with LLM")
    print("=" * 80)
    
    weave.init("molleo_demo_multi")
    
    class Args:
        max_objectives = ["qed"]  # Maximize drug-likeness
        min_objectives = ["sa"]   # Minimize synthetic accessibility
        mode = "weighted_sum"
        mol_lm = "GPT-4"
        model = "openai/gpt-4o-mini"
        population_size = 15
        offspring_size = 30
        generations = 2
        mutation_rate = 0.01
        initial_size = 4
        seed = 42
        weights = [1.0, 0.5]  # Weights for [qed, sa]
    
    args = Args()
    
    print(f"\nConfiguration:")
    print(f"- Maximize: {args.max_objectives}")
    print(f"- Minimize: {args.min_objectives}")
    print(f"- Model: {args.model}")
    print(f"- Generations: {args.generations}")
    
    results = run_multi_objective(args)
    
    print(f"\n✅ Multi-objective Optimization Complete!")
    print(f"   Best weighted score: {results['best_score']:.4f}")
    
    return results


def demo_custom_oracle():
    """Demo with custom oracle function"""
    print("\n" + "=" * 80)
    print("DEMO: Custom Oracle - Lipinski's Rule of 5")
    print("=" * 80)
    
    from src.oracles.base import MolecularOracle
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    class LipinskiOracle(MolecularOracle):
        """Oracle for Lipinski's Rule of 5 compliance"""
        
        def __init__(self):
            super().__init__("lipinski_score")
            
        def _evaluate_molecule_impl(self, smiles: str) -> float:
            """Score based on Lipinski's Rule of 5"""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
                
            # Calculate Lipinski descriptors
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Score based on compliance (1 point for each rule satisfied)
            score = 0.0
            if mw <= 500:
                score += 0.25
            if logp <= 5:
                score += 0.25
            if hbd <= 5:
                score += 0.25
            if hba <= 10:
                score += 0.25
                
            return score
    
    # Create optimizer with custom oracle
    oracle = LipinskiOracle()
    optimizer = MolLEOOptimizer(
        oracle=oracle,
        population_size=10,
        offspring_size=20,
        model_name="openai/gpt-4o-mini",
        use_llm_mutations=True
    )
    
    # Starting molecules
    initial = ["CCO", "c1ccccc1", "CC(=O)NC1=CC=C(C=C1)O"]
    
    print("\nRunning optimization for Lipinski compliance...")
    results = optimizer.optimize(initial, num_generations=2)
    
    print(f"\n✅ Custom Oracle Optimization Complete!")
    print(f"   Best Lipinski score: {results['best_score']:.4f}")
    print(f"   Best molecule: {results['best_molecule']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive MolLEO demo with SDE Harness"
    )
    parser.add_argument(
        "--mode", 
        choices=["single", "multi", "custom", "all"],
        default="all",
        help="Which demo to run"
    )
    args = parser.parse_args()
    
    print("\n🧪 MolLEO - Molecular Language-Enhanced Evolutionary Optimization")
    print("   Powered by SDE Harness Framework")
    
    if args.mode == "single" or args.mode == "all":
        demo_single_objective()
        
    if args.mode == "multi" or args.mode == "all":
        demo_multi_objective()
        
    if args.mode == "custom" or args.mode == "all":
        demo_custom_oracle()
    
    print("\n✨ All demos completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- ✅ LLM-guided molecular mutations")
    print("- ✅ Single and multi-objective optimization")
    print("- ✅ Custom oracle integration")
    print("- ✅ Full SDE Harness integration")
    print("- ✅ Experiment tracking with Weave")


if __name__ == "__main__":
    main()