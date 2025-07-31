#!/usr/bin/env python3
"""
BioDiscoveryAgent - AI agent for biological experiment design
Command Line Interface for gene discovery workflows.
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
from src.modes import run_perturb_genes, run_baseline, run_analyze
from src.utils.data_loader import validate_data_files


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="BioDiscoveryAgent - AI agent for biological experiment design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python cli.py perturb-genes --model openai/gpt-4o-2024-08-06 --run-name test-oracle --data-name IFNG --steps 5 --num-genes 128 --log-dir gpt4 --task-variant brief
  python cli.py baseline --data-name IFNG --sample-size 128  # Not implemented yet
  python cli.py analyze --dataset Carnevale22_Adenosine --model gpt-4o-2024-08-06
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="mode", help="Running mode")

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--log-dir", type=str, default="logs", help="Log directory (default: logs)")
    common_args.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    common_args.add_argument("--model", type=str, default="anthropic/claude-3-5-sonnet-20240620",
                           help="LLM model to use")
    common_args.add_argument("--temperature", type=float, default=0.1, help="Temperature for LLM (default: 0.1)")
    common_args.add_argument("--max-tokens", type=int, default=15000, help="Max tokens for LLM (default: 15000)")

    # Perturb genes mode (supports multiple task variants)
    perturb_parser = subparsers.add_parser(
        "perturb-genes",
        parents=[common_args],
        help="Run gene perturbation discovery"
    )
    perturb_parser.add_argument("--task-variant", type=str, default="brief",
                              choices=["brief", "full", "brief-NormanGI", "brief-Horlbeck"],
                              help="Task variant (default: brief)")
    perturb_parser.add_argument("--data-name", type=str, required=True,
                              help="Dataset name (e.g., IFNG, IL2)")
    perturb_parser.add_argument("--steps", type=int, default=5,
                              help="Number of discovery steps (default: 5)")
    perturb_parser.add_argument("--num-genes", type=int, default=128,
                              help="Number of genes to propose (default: 128)")
    perturb_parser.add_argument("--run-name", type=str, default="exp",
                              help="Run name for logging (default: exp)")
    perturb_parser.add_argument("--folder-name", type=str, default="temp",
                              help="Temp folder name (default: temp)")
    
    # Tool options
    perturb_parser.add_argument("--lit-review", action="store_true",
                              help="Enable literature review tool")
    perturb_parser.add_argument("--critique", action="store_true",
                              help="Enable AI critique tool")
    perturb_parser.add_argument("--gene-search", action="store_true",
                              help="Enable gene similarity search")
    perturb_parser.add_argument("--reactome", action="store_true",
                              help="Enable Reactome pathway search")
    perturb_parser.add_argument("--csv-path", type=str, default="./",
                              help="Path to CSV files for gene search")
    perturb_parser.add_argument("--use-oracle", action="store_true", default=True,
                              help="Use Oracle-based evaluator (default: True)")
    perturb_parser.add_argument("--no-oracle", dest="use_oracle", action="store_false",
                              help="Use standard evaluator instead of Oracle")
    
    # Baseline mode
    baseline_parser = subparsers.add_parser(
        "baseline",
        parents=[common_args],
        help="Run baseline sampling"
    )
    baseline_parser.add_argument("--data-name", type=str, required=True,
                               help="Dataset name")
    baseline_parser.add_argument("--sample-size", type=int, default=128,
                               help="Sample size (default: 128)")
    baseline_parser.add_argument("--pathways", nargs="+", default=[],
                               help="Pathways for sampling")
    
    # Analyze mode
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze results"
    )
    analyze_parser.add_argument("--dataset", type=str, required=True,
                              help="Dataset name")
    analyze_parser.add_argument("--model", type=str, required=True,
                              help="Model name (e.g., claude_all)")
    analyze_parser.add_argument("--rounds", type=int, default=5,
                              help="Number of rounds (default: 5)")
    analyze_parser.add_argument("--trials", type=int, default=1,
                              help="Number of trials (default: 1)")
    analyze_parser.add_argument("--essential", type=int, default=1,
                              help="Use essential genes filter (default: 1)")

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate mode
    if args.mode == "perturb-genes":
        # Validate data files
        if not validate_data_files(args.data_name):
            print(f"Error: Missing data files for dataset {args.data_name}")
            sys.exit(1)
        
        result = run_perturb_genes(args)
        print(f"\\nExperiment completed. Results saved to {args.log_dir}")
        
    elif args.mode == "baseline":
        if not validate_data_files(args.data_name):
            print(f"Error: Missing data files for dataset {args.data_name}")
            sys.exit(1)
            
        result = run_baseline(args)
        print(f"\\nBaseline sampling completed.")
        
    elif args.mode == "analyze":
        result = run_analyze(args)
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()