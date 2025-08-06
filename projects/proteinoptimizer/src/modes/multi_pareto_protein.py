"""Pareto-based multi-objective optimization for protein sequences."""
from __future__ import annotations

import os, sys
from typing import Dict, Any
import weave

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from ..oracles.protein_oracles import Syn3bfoOracle
from ..oracles.multi_objective_oracles import PottsObjective, HammingDistanceOracle
from ..core.pareto_optimizer import ParetoOptimizer


def run_multi_pareto(args) -> Dict[str, Any]:
    weave.init("proteinoptimizer_pareto_syn3bfo")

    # Setup oracles
    base_oracle = Syn3bfoOracle()
    if not base_oracle._potts_landscape:
        raise ValueError("Potts model NPZ file must be present for multi-objective mode.")
    potts_obj = PottsObjective(base_oracle._potts_landscape)
    wt_seq = "".join([base_oracle.ALPHABET[i] for i in base_oracle._potts_landscape.wildtype_sequence])
    hamming_obj = HammingDistanceOracle(reference_sequence=wt_seq)

    objectives = [potts_obj, hamming_obj] # no weights needed for pareto

    initial_population = base_oracle.get_initial_population(args.initial_size)

    optimizer = ParetoOptimizer(
        objectives=objectives,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        random_seed=args.seed,
        model_name=args.model,
        use_llm_mutations=bool(args.model),
    )

    results = optimizer.optimize(initial_population, num_generations=args.generations)
    
    # --- Reporting ---
    print("\n--- Pareto Front ---")
    pareto_front = results['pareto_front']
    print(f"Found {len(pareto_front)} non-dominated solutions.")
    for i, (seq, scores) in enumerate(pareto_front):
        print(f"{i+1}. Seq: {seq} | Scores: {scores}")
        
    return results 