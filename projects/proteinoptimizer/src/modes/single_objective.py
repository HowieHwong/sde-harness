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
        # Get initial sequences normally
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
    
    # Print summary
    print("\nOptimization Results:")
    print(f"Best sequence: {results['best_sequence']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Total oracle calls: {results['oracle_calls']}")
    
    # Save results
    if hasattr(args, 'output_dir') and args.output_dir:
        import json
        if args.model:
            model = args.model.split('/')[-1]
        else:
            model = 'baseline'
        output_file = os.path.join(
            args.output_dir, 
            f"results_single_{args.oracle}_{args.seed}_{model}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results