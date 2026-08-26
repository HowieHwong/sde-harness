#!/usr/bin/env python3
"""
LLMEO - LLM-based Evolutionary Optimization
Command Line Interface for scientific discovery workflows.
"""

import argparse
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

validate_data_files = None
run_few_shot = None
run_single_prop = None
run_multi_prop = None


def get_validate_data_files():
    """Load data validation only for commands that actually run a mode."""
    global validate_data_files

    if validate_data_files is None:
        from src.utils.data_loader import validate_data_files as loaded_validator
        validate_data_files = loaded_validator
    return validate_data_files


def get_mode_runner(mode: str):
    """Load mode implementations only after CLI parsing."""
    global run_few_shot, run_single_prop, run_multi_prop

    if mode == "few-shot":
        if run_few_shot is None:
            from src.modes import run_few_shot as loaded_runner
            run_few_shot = loaded_runner
        return run_few_shot

    if mode == "single-prop":
        if run_single_prop is None:
            from src.modes import run_single_prop as loaded_runner
            run_single_prop = loaded_runner
        return run_single_prop

    if mode == "multi-prop":
        if run_multi_prop is None:
            from src.modes import run_multi_prop as loaded_runner
            run_multi_prop = loaded_runner
        return run_multi_prop

    raise ValueError(f"Unknown mode: {mode}")


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
    common_args.add_argument("--model", type=str, default="deepseek/deepseek-v4-flash", help="Model name from the harness root models.yaml")
    common_args.add_argument("--temperature", type=float, default=1, help="Temperature")
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
    if not get_validate_data_files()():
        sys.exit(1)

    # Run corresponding mode
    try:
        runner = get_mode_runner(args.mode)

        if args.mode == "few-shot":
            print(args)
            runner(args)
        elif args.mode == "single-prop":
            runner(args)
        elif args.mode == "multi-prop":
            runner(args)
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
