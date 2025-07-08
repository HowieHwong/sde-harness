#!/usr/bin/env python3
"""
LLMEO - LLM-based Evolutionary Optimization
Command Line Interface for scientific discovery workflows.
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
from Projects.LLMEO.modes.structued_output_few_shot import run_few_shot_structured
from Projects.LLMEO.modes.structured_output_single_prop import run_single_prop_structured
from modes import run_few_shot, run_single_prop, run_multi_prop
from utils.data_loader import validate_data_files


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="LLMEO - LLM-based Evolutionary Optimization for Scientific Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python cli.py few-shot --iterations 3 --temperature 0.1
  python cli.py multi-prop --max-tokens 5000 --samples 20
  python cli.py single-prop --num-samples 15
  python cli.py diy-gen --temperature 0.8
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="mode", help="Running mode")

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--samples", type=int, default=10, help="Initial sample number (default: 10)")
    common_args.add_argument("--num-samples", type=int, default=10, help="Generated sample number (default: 10)")
    common_args.add_argument("--max-tokens", type=int, default=8000, help="Maximum token number (default: 8000)")
    common_args.add_argument("--iterations", type=int, default=2, help="Iteration number (default: 2)")
    common_args.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    common_args.add_argument("--model", type=str, default="deepseek/deepseek-chat", help="Choose From [openai/gpt-4o-2024-08-06, anthropic/claude-3-7-sonnet-20250219, deepseek/deepseek-chat]")
    common_args.add_argument("--temperature", type=float, default=1, help="Temperature")
    common_args.add_argument("--response-format", type=bool, default=False, help="Response format")
    # Few-shot mode
    few_shot_parser = subparsers.add_parser(
        "few-shot", parents=[common_args], help="Few-shot learning mode - Learning based on a few samples",
    )

    # Single-prop mode
    single_prop_parser = subparsers.add_parser("single-prop",parents=[common_args],help="Single-property optimization mode - Single optimization of specific properties",)

    # Multi-prop mode
    multi_prop_parser = subparsers.add_parser(
        "multi-prop",
        parents=[common_args],
        help="Multi-property optimization mode - Multi-round optimization of multiple properties",
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return


    # Validate data files
    if not validate_data_files():
        sys.exit(1)

    # Run corresponding mode
    try:
        if args.response_format == True and args.mode == "few-shot":
            print(args)
            run_few_shot_structured(args)
        if args.response_format == True and args.mode == "single-prop":
            print(args)
            run_single_prop_structured(args)

        if args.mode == "few-shot":
            print(args)
            run_few_shot(args)
        elif args.mode == "single-prop":
            run_single_prop(args)
        elif args.mode == "multi-prop":
            run_multi_prop(args)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  User interrupted execution")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
