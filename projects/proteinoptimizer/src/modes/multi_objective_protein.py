"""Multi-objective optimization (weighted sum) for protein sequences."""
from __future__ import annotations

import os, sys
from typing import Dict, Any, List
import weave

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from ..core import ProteinOptimizer
from ..oracles.protein_oracles import Syn3bfoOracle
from ..core.multiobjective import WeightedSumOracle
from ..oracles.multi_objective_oracles import PottsObjective, HammingDistanceOracle


def run_multi_objective(args) -> Dict[str, Any]:
    """Run weighted-sum multi-objective optimisation.

    Args:
        args should include:
            --potts-weight float
            --hamming-weight float
            --population-size, --offspring-size, etc.
    """
    weave.init("proteinoptimizer_multi_potts_hamming")

    # Setup oracles
    base_oracle = Syn3bfoOracle()
    if not base_oracle._potts_landscape:
        raise ValueError("Potts model NPZ file must be present for multi-objective mode.")
    potts_obj = PottsObjective(base_oracle._potts_landscape)
    wt_seq = "".join([base_oracle.ALPHABET[i] for i in base_oracle._potts_landscape.wildtype_sequence])
    hamming_obj = HammingDistanceOracle(reference_sequence=wt_seq)

    objectives = [
        (potts_obj, args.potts_weight),
        (hamming_obj, args.hamming_weight),
    ]

    combo_oracle = WeightedSumOracle(objectives)

    # Initial population
    initial_sequences = base_oracle.get_initial_population(args.initial_size)

    optimizer = ProteinOptimizer(
        oracle=combo_oracle,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        random_seed=args.seed,
    )

    results = optimizer.optimize(initial_sequences, num_generations=args.generations)

    print("\nMulti-objective Results:")
    print(f"Best seq: {results['best_sequence']}")
    print(f"Weighted score: {results['best_score']:.4f}")

    return results 