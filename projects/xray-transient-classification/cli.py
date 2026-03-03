#!/usr/bin/env python3
"""
X-Ray Transient Classification CLI

Command-line interface for classifying X-ray transients using LLMs
with physics-based evaluation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modes.classify import run_classify, add_classify_args
from src.utils.data_loader import list_transients, validate_data_files, get_data_dir


def cmd_classify(args: argparse.Namespace) -> int:
    """Run the classify command."""
    try:
        results = run_classify(args)
        
        if not args.output and not args.verbose:
            final = results.get('final_classification', {})
            print(f"Classification correct: {final.get('top1_correct', 0) == 1.0}")
        
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during classification: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List available transient observations."""
    transients = list_transients()
    
    if not transients:
        print("No transient observations found.")
        print(f"Expected location: {get_data_dir() / 'transients'}")
        return 1
    
    print(f"Available transients ({len(transients)}):")
    for t in transients:
        print(f"  - {Path(t).name}")
    
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate data files exist."""
    if validate_data_files():
        print("All required data files present.")
        return 0
    else:
        print("Missing required data files.", file=sys.stderr)
        print(f"Check data directory: {get_data_dir()}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='xray-transient-classification',
        description='Classify X-ray transients using LLMs with physics-based evaluation'
    )
    
    subparsers = parser.add_subparsers(
        title='commands',
        dest='command',
        help='Available commands'
    )
    
    classify_parser = subparsers.add_parser(
        'classify',
        help='Classify a single transient observation'
    )
    add_classify_args(classify_parser)
    classify_parser.set_defaults(func=cmd_classify)
    
    list_parser = subparsers.add_parser(
        'list',
        help='List available transient observations'
    )
    list_parser.set_defaults(func=cmd_list)
    
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate that required data files exist'
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
