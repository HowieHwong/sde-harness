"""Multi-objective optimization (weighted sum) for protein sequences."""
from __future__ import annotations

import os, sys
from typing import Dict, Any, List
import weave
import time

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from src.core import ProteinOptimizer
from src.core.multiobjective import WeightedSumOracle
from src.oracles import (
    Syn3bfoOracle, GB1Oracle, TrpBOracle, AAVOracle, GFPOracle,
    PottsObjective, HammingDistanceOracle
)


def run_multi_objective(args):
    """Run multi-objective optimization with weighted sum."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Select oracle
    if args.oracle == 'syn-3bfo':
        base_oracle = Syn3bfoOracle()
    elif args.oracle == 'gb1':
        base_oracle = GB1Oracle()
    elif args.oracle == 'trpb':
        base_oracle = TrpBOracle()
    elif args.oracle == 'aav':
        base_oracle = AAVOracle()
    elif args.oracle == 'gfp':
        base_oracle = GFPOracle()
    else:
        raise ValueError(f"Unknown oracle: {args.oracle}")

    # Weave setup
    experiment_name = f"protein_multiobj_{args.oracle}_{int(time.time())}"
    weave.init(project_name="sde-harness-protein_multiobj")

    print(f"Running multi-objective optimization for {args.oracle}...")

    # Define objectives. Use Potts if available, otherwise use the oracle's default fitness.
    # if base_oracle._potts_landscape is not None:
    #     print("Using Potts model for fitness objective.")
    #     fitness_objective = PottsObjective(base_oracle._potts_landscape)
    # else:
    print("Using CSV lookup for fitness objective.")
    fitness_objective = base_oracle

    # Get a wild-type sequence for Hamming distance calculation
    # This part needs a robust way to get a reference sequence.
    # For now, we'll try to get it from the oracle if the method exists,
    # otherwise we'll have to use a placeholder or raise an error.
    if hasattr(base_oracle, 'get_initial_population'):
        wt_sequence = base_oracle.get_initial_population(1)[0]
    else:
        # ML oracles might not have a dataset to sample from.
        # We need a defined reference sequence for them.
        # Using a placeholder for now. This should be improved.
        if args.oracle == 'aav':
            # Placeholder from observed data
            wt_sequence = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"
        elif args.oracle == 'gfp':
            # Placeholder from observed data
            wt_sequence = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        else:
            raise ValueError(f"Cannot determine a reference sequence for Hamming distance with oracle '{args.oracle}'")

    hamming_objective = HammingDistanceOracle(wt_sequence)

    # Combined oracle
    oracle = WeightedSumOracle(
        objectives=[
            (fitness_objective, args.fitness_weight),
            (hamming_objective, args.hamming_weight),
        ]
    )

    # This part also needs the reference sequences
    if hasattr(base_oracle, 'get_initial_population'):
        initial_sequences = base_oracle.get_initial_population(size=args.initial_size)
    else:
        # For ML oracles, we start with mutations of the wild-type
        initial_sequences = [wt_sequence] * args.initial_size

    optimizer = ProteinOptimizer(
        oracle=oracle,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        random_seed=args.seed,
        model_name=args.model,
        use_llm_mutations=bool(args.model),
    )

    results = optimizer.optimize(initial_sequences, num_generations=args.generations)

    print("\nMulti-objective Results:")
    print(f"Best seq: {results['best_sequence']}")
    print(f"Weighted score: {results['best_score']:.4f}")
    
    # Save results
    if hasattr(args, 'output_dir') and args.output_dir:
        import json
        if args.model:
            model = args.model.split('/')[-1]
        else:
            model = 'baseline'
        output_file = os.path.join(
            args.output_dir, 
            f"results_multi_{args.oracle}_{args.seed}_{model}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
    return results 