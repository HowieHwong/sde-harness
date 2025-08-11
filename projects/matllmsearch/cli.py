#!/usr/bin/env python3
"""
MatLLMSearch - LLM-based Crystal Structure Generation and Optimization
Command Line Interface for materials discovery workflows.
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
from src.modes import run_csg, run_csp, run_analyze
from src.utils.data_loader import validate_data_files


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="MatLLMSearch - LLM-based Crystal Structure Generation for Materials Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python cli.py csg --model meta-llama/Meta-Llama-3.1-70B-Instruct --population-size 100 --max-iter 5
  python cli.py csp --compound Ag6O2 --model meta-llama/Meta-Llama-3.1-8B-Instruct --population-size 50
  python cli.py analyze --results-path results/experiment_1
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="mode", help="Running mode")

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--log-dir", type=str, default="logs", help="Log directory (default: logs)")
    common_args.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    common_args.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct",
                           help="LLM model to use")
    common_args.add_argument("--temperature", type=float, default=1.0, help="Temperature for LLM (default: 1.0)")
    common_args.add_argument("--max-tokens", type=int, default=4000, help="Max tokens for LLM (default: 4000)")
    common_args.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size (default: 1)")
    common_args.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory utilization (default: 0.85)")

    # Crystal Structure Generation (CSG) mode
    csg_parser = subparsers.add_parser(
        "csg",
        parents=[common_args],
        help="Crystal Structure Generation - Generate novel crystal structures"
    )
    csg_parser.add_argument("--population-size", type=int, default=100,
                          help="Population size for genetic algorithm (default: 100)")
    csg_parser.add_argument("--reproduction-size", type=int, default=5,
                          help="Number of offspring per generation (default: 5)")
    csg_parser.add_argument("--context-size", type=int, default=5,
                          help="Context size for parent structures (default: 5)")
    csg_parser.add_argument("--max-iter", type=int, default=10,
                          help="Maximum iterations (default: 10)")
    csg_parser.add_argument("--pool-size", type=int, default=-1,
                          help="Reference pool size (-1 for all) (default: -1)")
    csg_parser.add_argument("--opt-goal", choices=["e_hull_distance", "bulk_modulus_relaxed", "multi-obj"], 
                          default="e_hull_distance", help="Optimization goal (default: e_hull_distance)")
    csg_parser.add_argument("--fmt", choices=["poscar", "cif"], default="poscar",
                          help="Structure format (default: poscar)")
    csg_parser.add_argument("--save-label", type=str, default="csg_experiment",
                          help="Experiment label for saving (default: csg_experiment)")
    csg_parser.add_argument("--resume", type=str, default="",
                          help="Resume from checkpoint directory")

    # Crystal Structure Prediction (CSP) mode
    csp_parser = subparsers.add_parser(
        "csp",
        parents=[common_args],
        help="Crystal Structure Prediction - Predict ground state structures"
    )
    csp_parser.add_argument("--compound", type=str, required=True,
                          choices=["Ag6O2", "Bi2F8", "Co2Sb2", "Co4B2", "Cr4Si4", "KZnF3", "Sr2O4", "YMg3"],
                          help="Target compound for structure prediction")
    csp_parser.add_argument("--population-size", type=int, default=100,
                          help="Population size (default: 100)")
    csp_parser.add_argument("--reproduction-size", type=int, default=5,
                          help="Number of offspring per generation (default: 5)")
    csp_parser.add_argument("--context-size", type=int, default=5,
                          help="Context size for parent structures (default: 5)")
    csp_parser.add_argument("--max-iter", type=int, default=20,
                          help="Maximum iterations (default: 20)")
    csp_parser.add_argument("--fmt", choices=["poscar", "cif"], default="poscar",
                          help="Structure format (default: poscar)")
    csp_parser.add_argument("--save-label", type=str, default="csp_experiment",
                          help="Experiment label for saving (default: csp_experiment)")

    # Analysis mode
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze experimental results"
    )
    analyze_parser.add_argument("--results-path", type=str, required=True,
                              help="Path to results directory")
    analyze_parser.add_argument("--experiment-name", type=str, default="experiment",
                              help="Experiment name (default: experiment)")

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate mode
    if args.mode == "csg":
        result = run_csg(args)
        print(f"\\nCSG experiment completed. Results saved to {args.log_dir}")
        
    elif args.mode == "csp":
        result = run_csp(args)
        print(f"\\nCSP experiment completed. Results saved to {args.log_dir}")
        
    elif args.mode == "analyze":
        result = run_analyze(args)
        print(f"\\nAnalysis completed.")
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()