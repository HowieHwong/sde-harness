"""Synchronous command-line interface for LLMSR equation discovery (symbolic regression)."""

import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path

# Patch: If sde_harness is not installed, add project root to sys.path
import sys
import os
import weave

# Ensure sde_harness is importable by adding project root to sys.path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sde_harness.base import CLIBase
from workflow import LLMSRWorkflowSync


class LLMSRCLISync(CLIBase):
    """Synchronous CLI for LLMSR equation discovery."""
    
    def __init__(self):
        super().__init__("LLMSR-Sync")
    
    def _add_project_arguments(self, parser: argparse.ArgumentParser):
        """Add LLMSR-specific arguments."""
        parser.add_argument(
            "--dataset",
            default="lsrtransform",
            choices=["lsrtransform", "bio_pop_growth", "chem_react", "matsci", "phys_osc"],
            help="Dataset to use for symbolic regression"
        )
        parser.add_argument(
            "--problem",
            type=str,
            help="Specific problem name to solve (optional)"
        )
        parser.add_argument(
            "--model",
            default="openai/gpt-4.1-nano-2025-04-14",
            help="LLM model to use for generation"
        )
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=5,
            help="Maximum number of optimization iterations"
        )
        parser.add_argument(
            "--max-params",
            type=int,
            default=10,
            help="Maximum number of parameters to optimize in the equation"
        )
        parser.add_argument(
            "--optimization-method",
            default="L-BFGS-B",
            choices=["BFGS", "L-BFGS-B", "SLSQP", "trust-constr"],
            help="Optimization method for equation parameter fitting"
        )
        parser.add_argument(
            "--problem-filter",
            nargs="+",
            help="List of specific problem names to process (optional)"
        )
        # parser.add_argument(
        #     "--output-dir",
        #     default="results",
        #     help="Output Directory"
        # )
        parser.add_argument("--project-name", 
            default="llm-sym-reg",
            help="Weave project name for logging")
    
    def run_command(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute the LLMSR command synchronously."""
        print("Starting LLMSR Equation Discovery (Synchronous)")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Max iterations: {args.max_iterations}")
        print(f"Output directory: {args.output_dir}")
        print(f"Project name: {args.project_name}")
    
        # Initialize weave logging
        weave.init(args.project_name)
        
        # Initialize workflow
        workflow = LLMSRWorkflowSync(
            model_name=args.model,
            max_iterations=args.max_iterations,
            max_params=args.max_params,
            optimization_method=args.optimization_method
        )
        
        # Setup dataset
        print(f"\n Loading dataset: {args.dataset}")
        workflow.setup_dataset(args.dataset)
        
        # Run discovery
        if args.problem:
            # Single problem
            print(f"\n Discovering equation for problem: {args.problem}")
            result = workflow.discover_equation_sync(args.problem, args.output_dir)
            results = {args.problem: result}
        else:
            # All problems or filtered problems
            print(f"\n Discovering equations for all problems")
            results = workflow.discover_all_equations_sync(
                output_dir=args.output_dir,
                problem_filter=args.problem_filter
            )
        
        # Print summary
        workflow.print_summary()
        
        return {
            "results": results,
            "workflow": workflow,
            "output_dir": args.output_dir
        }


def main():
    """Main entry point for the synchronous CLI."""
    cli = LLMSRCLISync()
    cli.main()


if __name__ == "__main__":
    main()

