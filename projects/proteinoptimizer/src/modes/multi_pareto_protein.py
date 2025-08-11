"""Pareto-based multi-objective optimization for protein sequences."""
from __future__ import annotations

import os, sys
from typing import Dict, Any
import weave
import time

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from src.core.pareto_optimizer import ParetoOptimizer
from src.oracles import (
    Syn3bfoOracle, GB1Oracle, TrpBOracle, AAVOracle, GFPOracle,
    PottsObjective, HammingDistanceOracle
)


def run_multi_pareto(args):
    """Run multi-objective optimization with Pareto selection."""
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
    weave.init(project_name="sde-harness-protein_pareto")

    print(f"Running Pareto optimization for {args.oracle}...")

    # Define objectives for the optimizer to use
    if base_oracle._potts_landscape is not None:
        print("Using Potts model for fitness objective.")
        fitness_objective = PottsObjective(base_oracle._potts_landscape)
    else:
        print("Using CSV lookup for fitness objective.")
        fitness_objective = base_oracle
        
    if hasattr(base_oracle, 'get_initial_population'):
        wt_sequence = base_oracle.get_initial_population(1)[0]
    else:
        if args.oracle == 'aav':
            wt_sequence = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"
        elif args.oracle == 'gfp':
            wt_sequence = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        else:
            raise ValueError(f"Cannot determine a reference sequence for Hamming distance with oracle '{args.oracle}'")

    hamming_objective = HammingDistanceOracle(wt_sequence)
    objectives = [fitness_objective, hamming_objective]

    if hasattr(base_oracle, 'get_initial_population'):
        initial_sequences = base_oracle.get_initial_population(size=args.initial_size)
    else:
        initial_sequences = [wt_sequence] * args.initial_size

    optimizer = ParetoOptimizer(
        objectives=objectives,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        random_seed=args.seed,
        model_name=args.model,
        use_llm_mutations=bool(args.model),
    )

    results = optimizer.optimize(initial_sequences, num_generations=args.generations)
    
    # --- Reporting ---
    print("\n--- Pareto Front ---")
    pareto_front = results['pareto_front']
    print(f"Found {len(pareto_front)} non-dominated solutions.")
    for i, (seq, scores) in enumerate(pareto_front):
        print(f"{i+1}. Seq: {seq} | Scores: {scores}")
        
    return results 