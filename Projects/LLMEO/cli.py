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
from modes import run_few_shot, run_single_prop, run_multi_prop
from utils.data_loader import validate_data_files
from config.settings import get_default_config, validate_config


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
    common_args.add_argument("--api-key", type=str, help="OpenAI API key")
    common_args.add_argument(
        "--samples", type=int, default=10, help="Initial sample number (default: 10)"
    )
    common_args.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Generated sample number (default: 10)",
    )
    common_args.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Maximum token number (default: 5000)",
    )
    common_args.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature parameter (default: 0.0)",
    )
    common_args.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Few-shot mode
    few_shot_parser = subparsers.add_parser(
        "few-shot",
        parents=[common_args],
        help="Few-shot learning mode - Learning based on a few samples",
    )
    few_shot_parser.add_argument(
        "--iterations", type=int, default=2, help="Iteration number (default: 2)"
    )

    # Single-prop mode
    single_prop_parser = subparsers.add_parser(
        "single-prop",
        parents=[common_args],
        help="Single-property optimization mode - Single optimization of specific properties",
    )

    # Multi-prop mode
    multi_prop_parser = subparsers.add_parser(
        "multi-prop",
        parents=[common_args],
        help="Multi-property optimization mode - Multi-round optimization of multiple properties",
    )
    multi_prop_parser.add_argument(
        "--iterations", type=int, default=2, help="Iteration number (default: 2)"
    )

    # DIY generation mode
    # diy_gen_parser = subparsers.add_parser(
    #     'diy-gen',
    #     parents=[common_args],
    #     help='DIY generation mode - Custom generation parameters'
    # )

    # Parse arguments
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    # Check API key
    if not args.api_key and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OpenAI API key is required")
        print("Please set environment variable: export OPENAI_API_KEY='your-api-key'")
        print("Or use parameter: --api-key 'your-api-key'")
        sys.exit(1)

    # Validate data files
    if not validate_data_files():
        sys.exit(1)

    # Validate configuration
    config = get_default_config()
    if not validate_config(config):
        print("❌ Configuration validation failed")
        sys.exit(1)

    # Run corresponding mode
    try:
        if args.mode == "few-shot":
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
