#!/usr/bin/env python3
"""
MatLLMSearch - LLM-based Crystal Structure Generation and Optimization
Command Line Interface for materials discovery workflows.
"""

import argparse
import sys
import os
from pathlib import Path
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
  python cli.py analyze --input data/llama_test.csv --output evaluation_results.json
  python cli.py analyze --results-path results/experiment_1  # Uses generations.csv from results path
  python cli.py analyze --generate --num-structures 10 --model openai/gpt-5-mini --output api_results.json # Generate via API
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="mode", help="Running mode")

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--log-dir", type=str, default="logs", help="Log directory (default: logs)")
    common_args.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    common_args.add_argument("--data-path", type=str, default="data/band_gap_processed_5000.csv", help="Path to seed structures data file (default: data/band_gap_processed_5000.csv)")
    common_args.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct",
                           help="LLM model to use")
    common_args.add_argument("--temperature", type=float, default=1.0, help="Temperature for LLM (default: 1.0)")
    common_args.add_argument("--max-tokens", type=int, default=4000, help="Max tokens for LLM (default: 4000)")

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
    csg_parser.add_argument("--parent-size", type=int, default=2,
                          help="Number of parent structures per group (default: 2)")
    csg_parser.add_argument("--max-iter", type=int, default=10,
                          help="Maximum iterations (default: 10)")
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
    csp_parser.add_argument("--parent-size", type=int, default=2,
                          help="Number of parent structures per group (default: 2)")
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
    analyze_parser.add_argument("--input", type=str, default=None,
                           help="Input CSV file with structures (default: data/llama_test.csv)")
    analyze_parser.add_argument("--results-path", type=str, default=None,
                              help="Path to results directory (alternative to --input, looks for generations.csv)")
    analyze_parser.add_argument("--generate", action="store_true",
                           help="Generate structures via API instead of reading from file")
    analyze_parser.add_argument("--model", type=str, default="openai/gpt-4o-mini",
                           help="Model to use for API generation (default: openai/gpt-4o-mini)")
    analyze_parser.add_argument("--temperature", type=float, default=1.0,
                           help="Temperature for generation (default: 1.0)")
    analyze_parser.add_argument("--max-tokens", type=int, default=4000,
                           help="Max tokens for generation (default: 4000)")
    analyze_parser.add_argument("--fmt", choices=["poscar", "cif"], default="poscar",
                           help="Structure format for generation (default: poscar)")
    analyze_parser.add_argument("--data-path", type=str, default="data/band_gap_processed_5000.csv",
                           help="Path to seed structures data file for reference pool (default: data/band_gap_processed_5000.csv)")
    analyze_parser.add_argument("--max-iter", type=int, default=1,
                           help="Maximum iterations for generation (default: 1)")
    analyze_parser.add_argument("--population-size", type=int, default=None,
                           help="Population size for generation (default: same as --num-structures)")
    analyze_parser.add_argument("--reproduction-size", type=int, default=5,
                           help="Number of offspring per generation (default: 5)")
    analyze_parser.add_argument("--parent-size", type=int, default=2,
                           help="Number of parent structures per group (default: 2)")
    analyze_parser.add_argument("--seed", type=int, default=42,
                           help="Random seed (default: 42)")
    analyze_parser.add_argument("--training-data", type=str, default=None,
                           help="Training data file for novelty calculation (default: data/mp_20/train.csv)")
    analyze_parser.add_argument("--output", type=str, default=None,
                           help="Output JSON file for results")
    analyze_parser.add_argument("--experiment-name", type=str, default="experiment",
                              help="Experiment name (used when --results-path is specified)")
    analyze_parser.add_argument("--limit", type=int, default=0,
                           help="Limit number of structures to evaluate (0 = all, for faster testing)")
    analyze_parser.add_argument("--mlip", type=str, default="chgnet",
                           help="Machine learning interatomic potential (default: chgnet)")
    analyze_parser.add_argument("--ppd-path", type=str, default="data/2023-02-07-ppd-mp.pkl.gz",
                           help="Path to patched phase diagram file")
    analyze_parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                           help="Device for computation (default: cuda)")

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