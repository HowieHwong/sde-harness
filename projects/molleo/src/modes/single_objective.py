"""Single objective optimization mode for MolLEO"""

import sys
import os
from typing import Dict, Any, List
import weave

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from sde_harness.core import Workflow
from ..core import MolLEOOptimizer
from ..oracles import TDCOracle


def run_single_objective(args) -> Dict[str, Any]:
    """
    Run single objective optimization
    
    Args:
        args: Command line arguments with:
            - oracle: Oracle name (e.g., 'jnk3', 'qed', 'sa')
            - model: LLM model name
            - population_size: Population size
            - offspring_size: Offspring size per generation
            - generations: Number of generations
            - mutation_rate: Mutation probability
            - seed: Random seed
            - initial_size: Number of initial molecules
            
    Returns:
        Optimization results
    """
    print(f"🚀 Running single objective optimization for {args.oracle}...")
    
    # Initialize Weave for tracking
    weave.init(f"molleo_single_{args.oracle}")
    
    # Create oracle
    oracle = TDCOracle(args.oracle)
    
    # Get initial molecules
    initial_smiles = []
    
    # Sample initial molecules
    if hasattr(args, 'initial_smiles') and args.initial_smiles:
        initial_smiles = args.initial_smiles
    else:
        # Use some common drug molecules as starting points
        print("Using default initial molecules...")
        default_molecules = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC(=O)OC1=CC=CC=C1C(=O)O",        # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # Caffeine
            "CC(C)NCC(COC1=CC=CC=C1)O",        # Propranolol
            "CN1CCC(CC1)C2=C(OC3=CC=CC=C23)C4=CC=CC=C4",  # Tamoxifen
            "CC(=O)NC1=CC=C(C=C1)O",           # Paracetamol
            "CC1=C(C=C(C=C1)C(C)C)C(C)C",      # p-Cymene
            "O=C(O)C1=CC=CC=C1O",               # Salicylic acid
            "CCC(C)C1CCC(CC1)C(C)CC",          # Menthol derivative
            "CC1=CC=C(C=C1)C(C)(C)C",          # tert-Butyltoluene
        ]
        
        # If TDC is available, try to get some molecules from it
        try:
            from tdc.generation import MolGen
            # Use a valid dataset like 'zinc' or 'moses'
            data = MolGen(name='zinc')
            # Get some molecules
            df = data.get_data()
            if len(df) > 0:
                # Add some molecules from TDC
                default_molecules = df.sample(n=min(10, len(df)))['smiles'].tolist()
                print(f"Successfully loaded {len(default_molecules)} molecules from TDC")
        except Exception as e:
            print(f"TDC sampling failed: {e}")
            print("Using default molecules...")
            
        initial_smiles = default_molecules[:args.initial_size]
        
    print(f"Starting with {len(initial_smiles)} initial molecules")
    
    # Create optimizer
    optimizer = MolLEOOptimizer(
        oracle=oracle,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        n_jobs=args.n_jobs if hasattr(args, 'n_jobs') else -1,
        model_name=args.model,
        use_llm_mutations=bool(args.model) if hasattr(args, 'model') else False
    )
    
    # Run optimization
    results = optimizer.optimize(
        starting_smiles=initial_smiles,
        num_generations=args.generations
    )
    
    # Print summary
    print("\n📊 Optimization Results:")
    print(f"Best molecule: {results['best_molecule']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Total oracle calls: {results['oracle_calls']}")
    
    # Save results
    if hasattr(args, 'output_dir') and args.output_dir:
        import json
        output_file = os.path.join(
            args.output_dir, 
            f"results_{args.oracle}_{args.model.replace('/', '_') if args.model else 'random'}_{args.seed}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results