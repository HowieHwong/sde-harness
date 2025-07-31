#!/usr/bin/env python3
"""
Backward compatibility wrapper for analyze.py
Maps old command line arguments to new CLI structure.
"""
import sys
import argparse


def main():
    """Main compatibility wrapper."""
    # Parse original arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--essential', type=int, default=1)
    
    args = parser.parse_args()
    
    # Build new CLI command
    new_argv = [
        "cli.py",
        "analyze",
        "--dataset", args.dataset,
        "--model", args.model,
        "--rounds", str(args.rounds),
        "--trials", str(args.trials),
        "--essential", str(args.essential)
    ]
    
    # Replace sys.argv and import cli
    sys.argv = new_argv
    
    # Import and run CLI
    from cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()