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

    # Check if resuming from previous results
    resume_data = None
    initial_population_with_scores = None
    start_generation = 0
    num_generations = args.generations
    
    if hasattr(args, 'resume_results') and args.resume_results:
        import json
        resume_file = args.resume_results
        if not os.path.exists(resume_file):
            raise FileNotFoundError(f"Resume results file not found: {resume_file}")
        
        print(f"Loading resume data from: {resume_file}")
        with open(resume_file, 'r') as f:
            resume_data = json.load(f)
        
        # Extract final_population as [(sequence, score), ...]
        if 'final_population' in resume_data:
            final_pop = resume_data['final_population']
            if isinstance(final_pop[0], list) and len(final_pop[0]) == 2:
                # Format: [["sequence", score], ...]
                initial_population_with_scores = [(item[0], float(item[1])) for item in final_pop]
            else:
                raise ValueError("Resume file final_population format not recognized")
        else:
            raise ValueError("Resume results file missing 'final_population' field")
        
        # Determine how many generations to load
        if hasattr(args, 'continue_generations') and args.continue_generations is not None:
            num_generations_to_load = args.continue_generations - 1
            num_generations = args.continue_generations
        else:
            num_generations_to_load = None
        
        # Load only the specified number of generations from history (for display only)
        if num_generations_to_load is not None and num_generations_to_load > 0:
            best_scores_history = resume_data.get('best_scores_history', [])
            if len(best_scores_history) > num_generations_to_load:
                print(f"Loading only first {num_generations_to_load} generations from resume file (file has {len(best_scores_history)} generations)")
                start_generation = num_generations_to_load
            else:
                print(f"Resume file has {len(best_scores_history)} generations, using all")
                start_generation = len(best_scores_history)
        else:
            best_scores_history = resume_data.get('best_scores_history', [])
            start_generation = len(best_scores_history)
        
        # Display best score from resume (for info only)
        if 'best_score' in resume_data:
            print(f"Previous best score: {resume_data['best_score']:.4f} (for display only)")
        
        print(f"Resuming from generation {start_generation}")
        print(f"Loaded {len(initial_population_with_scores)} sequences from final population")
        print(f"Will continue for {num_generations} more generations")
    else:
        # This part also needs the reference sequences
        if hasattr(base_oracle, 'get_initial_population'):
            initial_sequences = base_oracle.get_initial_population(size=args.initial_size)
        else:
            # For ML oracles, we start with mutations of the wild-type
            initial_sequences = [wt_sequence] * args.initial_size
        print(f"Starting with {len(initial_sequences)} initial sequences")

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
    if resume_data and initial_population_with_scores is not None:
        results = optimizer.optimize(
            starting_sequences=[],  # Not used when initial_population_with_scores is provided
            num_generations=num_generations,
            initial_population_with_scores=initial_population_with_scores,
            start_generation=start_generation,
        )
    else:
        results = optimizer.optimize(
            starting_sequences=initial_sequences,
            num_generations=num_generations,
        )

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