#!/usr/bin/env python3
"""
ProteinOptimizer CLI – evolutionary optimization for protein sequences (Syn-3bfo)
"""

import argparse
import sys
import os
from typing import Optional

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="ProteinOptimizer – evolutionary optimization for protein sequences (Syn-3bfo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Optimize Syn-3bfo fitness landscape for 10 generations
  python cli.py single --generations 10 --population-size 100 --offspring-size 200
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="mode", help="Optimization mode")

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-2024-08-06",
        help="Model name for LLM-guided mutations (e.g., openai/gpt-4o-2024-08-06, claude-3-opus-20240229). Use 'none' for random mutations only."
    )
    common_args.add_argument(
        "--population-size", 
        type=int, 
        default=10,
        help="Population size (default: 10)"
    )
    common_args.add_argument(
        "--offspring-size",
        type=int,
        default=20,
        help="Offspring per generation (default: 20)"
    )
    common_args.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of generations (default: 20)"
    )
    common_args.add_argument(
        "--mutation-rate",
        type=float,
        default=0.01,
        help="Mutation probability (default: 0.01)"
    )
    common_args.add_argument(
        "--initial-size",
        type=int,
        default=20,
        help="Number of initial sequences (default: 20)"
    )
    common_args.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[0],
        help="Random seed(s) (default: 0)"
    )
    common_args.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )

    # Oracle choices
    oracle_choices = ['syn-3bfo', 'gb1', 'trpb', 'aav', 'gfp']

    # Single objective mode (inherits common args)
    single_parser = subparsers.add_parser(
        "single",
        parents=[common_args],
        help="Run single objective optimization",
    )
    single_parser.add_argument(
        "--oracle",
        type=str,
        default="syn-3bfo",
        choices=oracle_choices,
        help="Oracle to use for optimization",
    )

    # Multi-objective weighted-sum mode
    multi_parser = subparsers.add_parser(
        "multi",
        parents=[common_args],
        help="Multi-objective optimization (weighted sum)",
    )
    multi_parser.add_argument(
        "--oracle",
        type=str,
        default="syn-3bfo",
        choices=oracle_choices,
        help="Oracle to use for optimization.",
    )
    multi_parser.add_argument("--fitness-weight", type=float, default=1.0, help="Weight for fitness/Potts energy (maximise)")
    multi_parser.add_argument("--hamming-weight", type=float, default=-0.1, help="Weight for Hamming distance (minimise)")

    # Pareto mode
    pareto_parser = subparsers.add_parser(
        "multi-pareto",
        parents=[common_args],
        help="Multi-objective optimization with Pareto selection",
    )
    pareto_parser.add_argument(
        "--oracle",
        type=str,
        default="syn-3bfo",
        choices=oracle_choices,
        help="Oracle to use for optimization.",
    )
    # No specific args for pareto, it will use potts + hamming

    # Workflow mode
    workflow_parser = subparsers.add_parser(
        "workflow",
        parents=[common_args],
        help="Run GA inside SDE-Harness Workflow",
    )
    workflow_parser.add_argument(
        "--oracle",
        type=str,
        default="syn-3bfo",
        choices=oracle_choices,
        help="Oracle to use for the workflow",
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    # Convert 'none' to None for no LLM
    if args.model and args.model.lower() == 'none':
        args.model = None

    # Ensure seeds is iterable
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]

    for seed in seeds:
        args.seed = seed  # Set the current seed
        print(f"\n{'='*60}")
        print(f"Running with seed {seed}")
        print(f"{'='*60}\n")

        try:
            if args.mode == "single":
                from src.modes.single_objective import run_single_objective
                run_single_objective(args)
            elif args.mode == "multi":
                from src.modes.multi_objective_protein import run_multi_objective
                # Rename potts_weight to fitness_weight for the function call
                if 'potts_weight' in args:
                    args.fitness_weight = args.potts_weight
                run_multi_objective(args)
            elif args.mode == "multi-pareto":
                from src.modes.multi_pareto_protein import run_multi_pareto
                run_multi_pareto(args)
            elif args.mode == "workflow":
                from src.workflow import ProteinWorkflow
                import asyncio
                # Extract GA params from common args
                ga_params = {
                    k: v for k, v in vars(args).items()
                    if k in ["population_size", "offspring_size", "generations", "mutation_rate", "initial_size"]
                }
                flow = ProteinWorkflow(oracle_name=args.oracle, model_name=args.model, **ga_params)
                # Dummy prompt, workflow runs GA
                from sde_harness.core import Prompt
                prompt = Prompt(template_name="protein_opt", default_vars={"task":"N/A", "seq_len":0, "score1":0, "score2":0, "protein_seq_1":"-", "protein_seq_2":"-", "mean":0, "std":0})
                result = asyncio.run(flow.run(prompt=prompt, reference=None, gen_args={"model": args.model}))
                print("Workflow finished. Best sequence:", result["best_iteration"]["output"])
            else:
                print(f"Unknown mode: {args.mode}")
                sys.exit(1)

        except KeyboardInterrupt:
            print("\n User interrupted execution")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()