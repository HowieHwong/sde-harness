"""Single objective optimization mode for protein sequences (Syn-3bfo)"""

import sys
import os
import time
from typing import Dict, Any, List
import weave

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

# No Workflow needed for sequence GA
from src.core import ProteinOptimizer
from src.oracles import Syn3bfoOracle, GB1Oracle, TrpBOracle, AAVOracle, GFPOracle


def run_single_objective(args) -> Dict[str, Any]:
    """
    Run single objective optimization
    
    Args:
        args: Command line arguments with:
            - oracle: Should be 'syn-3bfo'
            - population_size: Population size
            - offspring_size: Offspring size per generation
            - generations: Number of generations
            - mutation_rate: Mutation probability
            - seed: Random seed
            - initial_size: Number of initial sequences
            
    Returns:
        Optimization results
    """
    # Weave setup
    weave.init(project_name="sde-harness-protein_singleobj")

    print(f"Running single objective optimization for {args.oracle}...")
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select oracle
    if args.oracle == 'syn-3bfo':
        oracle = Syn3bfoOracle()
    elif args.oracle == 'gb1':
        oracle = GB1Oracle()
    elif args.oracle == 'trpb':
        oracle = TrpBOracle()
    elif args.oracle == 'aav':
        oracle = AAVOracle()
    elif args.oracle == 'gfp':
        oracle = GFPOracle()
    else:
        raise ValueError(f"Unknown oracle: {args.oracle}")

    # Get initial sequences
    if hasattr(oracle, 'get_initial_population'):
        initial_sequences = oracle.get_initial_population(args.initial_size)
    else:
        # Fallback for ML oracles without a dataset to sample from
        if args.oracle == 'aav':
            wt_sequence = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"
        elif args.oracle == 'gfp':
            wt_sequence = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        else:
            raise ValueError(f"Cannot generate initial population for oracle '{args.oracle}'")
        initial_sequences = [wt_sequence] * args.initial_size

    print(f"Starting with {len(initial_sequences)} initial sequences")

    # Create optimizer
    optimizer = ProteinOptimizer(
        oracle=oracle,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        random_seed=args.seed,
        model_name=args.model,
        use_llm_mutations=bool(args.model),
    )
    
    # Run optimization
    results = optimizer.optimize(
        starting_sequences=initial_sequences,
        num_generations=args.generations,
    )
    
    # Print summary
    print("\nOptimization Results:")
    print(f"Best sequence: {results['best_sequence']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Total oracle calls: {results['oracle_calls']}")
    
    # Save results
    if hasattr(args, 'output_dir') and args.output_dir:
        import json
        output_file = os.path.join(
            args.output_dir, 
            f"results_{args.oracle}_{args.seed}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results