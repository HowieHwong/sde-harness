#!/usr/bin/env python3
"""
MolLEO - Molecular Language-Enhanced Evolutionary Optimization
Command Line Interface using SDE-Harness framework
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

# Import local modules
from src.modes import run_single_objective, run_multi_objective


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="MolLEO - LLM-augmented evolutionary algorithm for molecular discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Single objective optimization with LLM
  python cli.py single --oracle drd2 --model openai/gpt-4o-2024-08-06 --generations 20
  
  # Single objective optimization without LLM (random mutations only)
  python cli.py single --oracle qed --model none --generations 20
  
  # Multi-objective optimization
  python cli.py multi --max-obj logp qed --min-obj sa --model openai/gpt-4o-2024-08-06
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
        help="Population size (default: 100)"
    )
    common_args.add_argument(
        "--offspring-size",
        type=int,
        default=20,
        help="Offspring per generation (default: 200)"
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
        help="Number of initial molecules (default: 20)"
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

    # Single objective mode
    single_parser = subparsers.add_parser(
        "single",
        parents=[common_args],
        help="Single objective optimization"
    )
    single_parser.add_argument(
        "--oracle",
        type=str,
        required=True,
        choices=["jnk3", "gsk3b", "drd2", "qed", "sa", "logp", "penalized_logp"],
        help="Oracle function to optimize"
    )

    # Multi-objective mode (weighted sum)
    multi_parser = subparsers.add_parser(
        "multi",
        parents=[common_args],
        help="Multi-objective optimization with weighted sum"
    )
    multi_parser.add_argument(
        "--max-obj",
        type=str,
        nargs="+",
        required=True,
        help="Objectives to maximize"
    )
    multi_parser.add_argument(
        "--min-obj",
        type=str,
        nargs="+",
        default=[],
        help="Objectives to minimize"
    )
    multi_parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Weights for objectives (default: equal weights)"
    )

    # Multi-objective Pareto mode
    pareto_parser = subparsers.add_parser(
        "multi-pareto",
        parents=[common_args],
        help="Multi-objective optimization with Pareto selection"
    )
    pareto_parser.add_argument(
        "--max-obj",
        type=str,
        nargs="+",
        required=True,
        help="Objectives to maximize"
    )
    pareto_parser.add_argument(
        "--min-obj",
        type=str,
        nargs="+",
        default=[],
        help="Objectives to minimize"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    # Convert 'none' to None for no LLM
    if args.model and args.model.lower() == 'none':
        args.model = None

    # Run for each seed
    seeds = args.seed  # Save the list
    for seed in seeds:
        args.seed = seed  # Set the current seed
        print(f"\n{'='*60}")
        print(f"Running with seed {seed}")
        print(f"{'='*60}\n")

        try:
            if args.mode == "single":
                run_single_objective(args)
            elif args.mode == "multi":
                args.mode = "weighted_sum"
                args.max_objectives = args.max_obj
                args.min_objectives = args.min_obj
                # Set default equal weights if not provided
                if not hasattr(args, 'weights') or args.weights is None:
                    total_objectives = len(args.max_obj) + len(args.min_obj)
                    args.weights = [1.0] * total_objectives
                run_multi_objective(args)
            elif args.mode == "multi-pareto":
                args.mode = "pareto"
                args.max_objectives = args.max_obj
                args.min_objectives = args.min_obj
                # Set default equal weights if not provided
                if not hasattr(args, 'weights') or args.weights is None:
                    total_objectives = len(args.max_obj) + len(args.min_obj)
                    args.weights = [1.0] * total_objectives
                run_multi_objective(args)
            else:
                print(f"❌ Unknown mode: {args.mode}")
                sys.exit(1)

        except KeyboardInterrupt:
            print("\n⏹️  User interrupted execution")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()