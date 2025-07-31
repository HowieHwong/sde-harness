#!/usr/bin/env python3
"""
Example usage of MolLEO without LLM calls (using only random mutations)
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Add the molleo project root to path
molleo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, molleo_root)

from src.core import MolLEOOptimizer
from src.oracles import TDCOracle


def main():
    """Run MolLEO without LLM for testing"""
    print("=" * 60)
    print("MolLEO Example - QED Optimization (No LLM)")
    print("=" * 60)
    
    # Create oracle for drug-likeness (QED)
    oracle = TDCOracle("qed")
    
    # Create optimizer without LLM
    optimizer = MolLEOOptimizer(
        oracle=oracle,
        population_size=50,
        offspring_size=100,
        mutation_rate=0.02,  # Higher mutation rate since no LLM guidance
        use_llm_mutations=False
    )
    
    # Starting molecules - diverse drug-like compounds
    initial_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",           # Ibuprofen
        "CC(=O)OC1=CC=CC=C1C(=O)O",                # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",            # Caffeine
        "CC(C)NCC(COC1=CC=CC=C1)O",                # Propranolol
        "CN1CCC(CC1)C2=C(OC3=CC=CC=C23)C4=CC=CC=C4",  # Tamoxifen
        "CC(=O)NC1=CC=C(C=C1)O",                   # Paracetamol
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",     # Testosterone
        "CC(C)(C)NCC(COC1=CC=CC2=C1CCCC2=O)O",     # Levobunolol
        "CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2",       # Diphenhydramine
        "CC1=C(C=C(C=C1)C(C)C)C(C)(C)C",           # BHT
    ]
    
    print(f"\nStarting optimization with {len(initial_smiles)} molecules")
    print("Target: Maximize QED (drug-likeness)")
    print("Using: Random mutations only (no LLM)\n")
    
    # Run optimization
    results = optimizer.optimize(
        starting_smiles=initial_smiles,
        num_generations=10
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nBest molecule found: {results['best_molecule']}")
    print(f"Best QED score: {results['best_score']:.4f}")
    print(f"Total molecules evaluated: {len(results['all_results'])}")
    
    # Show top 5 molecules
    print("\nTop 5 molecules:")
    sorted_molecules = sorted(
        results['all_results'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    for i, (smiles, score) in enumerate(sorted_molecules, 1):
        print(f"{i}. {smiles}")
        print(f"   QED: {score:.4f}")
    
    # Show score progression
    print("\nScore progression:")
    for i in range(0, len(results['best_scores_history']), 2):
        gen = i + 1
        score = results['best_scores_history'][i]
        print(f"  Generation {gen:2d}: {score:.4f}")
    
    # Evaluate a known drug-like molecule for comparison
    print("\nReference molecules:")
    reference_drugs = {
        "Metformin": "CN(C)C(=N)NC(=N)N",
        "Atorvastatin": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
        "Omeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC",
    }
    
    for name, smiles in reference_drugs.items():
        try:
            score = oracle.evaluate_molecule(smiles)
            print(f"  {name}: QED = {score:.4f}")
        except:
            print(f"  {name}: Could not evaluate")


if __name__ == "__main__":
    main()