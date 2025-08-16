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
from workflow_evolutionary import LLMSREvolutionaryWorkflow
from workflow_llmsr_evolutionary import LLMSROriginalEvolutionaryWorkflow


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


        # Evolutionary parameters
        parser.add_argument(
            "--mode",
            type=str,
            choices=["evolutionary", "iterative", "evolutionary-original"],
            default="sync",
            help="Mode to run the workflow in (default: sync)"
        )
        parser.add_argument(
            "--num-islands", 
            type=int, 
            default=10,
            help="Number of islands in evolutionary buffer (default: 4)"
        )
        parser.add_argument(
            "--max-programs-per-island", 
            type=int, 
            default=50,
            help="Maximum programs per island (default: 50)"
        )
        parser.add_argument(
            "--reset-period", 
            type=int, 
            default=100,
            help="How often to reset weaker islands (default: 100)"
        )
        parser.add_argument(
            "--reset-fraction", 
            type=float, 
            default=0.5,
            help="Fraction of islands to reset (default: 0.5)"
        )
        parser.add_argument(
            "--num-examples", 
            type=int, 
            default=2,
            help="Number of examples to include in prompts (default: 3)"
        )
        parser.add_argument(
            "--example-temperature", 
            type=float, 
            default=1.0,
            help="Temperature for Boltzmann sampling (default: 1.0)"
        )
        parser.add_argument(
            "--example-temperature-period", 
            type=int, 
            default=100,
            help="Period for temperature annealing (default: 100)"
        )
        parser.add_argument(
            "--sampling-strategy", 
            type=str, 
            choices=["boltzmann", "best", "random"],
            default="boltzmann",
            help="Sampling strategy for examples (default: boltzmann)"
        )
    
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
        if args.mode == "evolutionary-original":
            workflow = LLMSROriginalEvolutionaryWorkflow(
                model_name=args.model,
                max_iterations=args.max_iterations,
                max_params=args.max_params,
                optimization_method=args.optimization_method,
                num_islands=args.num_islands,
                functions_per_prompt=args.num_examples,
                cluster_sampling_temperature_init=args.example_temperature,
                cluster_sampling_temperature_period=args.example_temperature_period,
                reset_period=args.reset_period
            )

        elif args.mode == "evolutionary":
            workflow = LLMSREvolutionaryWorkflow(
                model_name=args.model,
                max_iterations=args.max_iterations,
                max_params=args.max_params,
                optimization_method=args.optimization_method,
                num_islands=args.num_islands,
                max_programs_per_island=args.max_programs_per_island,
                reset_period=args.reset_period,
                reset_fraction=args.reset_fraction,
                num_examples=args.num_examples,
                temperature=args.example_temperature,
                sampling_strategy=args.sampling_strategy
            )
        else:
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
            if args.mode == "evolutionary-original":
                result = workflow.discover_equation_llmsr_evolutionary(args.problem, args.output_dir)
            elif args.mode == "evolutionary":
                result = workflow.discover_equation_evolutionary(args.problem, args.output_dir) 
            else:
                print(f"\n Discovering equation for problem: {args.problem}")
                result = workflow.discover_equation_sync(args.problem, args.output_dir)
            # Print best result
            best_equation = workflow.get_best_equation(args.problem)
            if best_equation:
                print(f"Best NMSE: {best_equation['nmse_score']:.6f}")
                print(f"Best Equation: {best_equation['equation_code']:.6f}")
                print(f"Found in iteration: {best_equation['iteration']}")
            
            results = {args.problem: result}
        
        else:
            # All problems or filtered problems
            print(f"\n Discovering equations for all problems")
            if args.mode == "evolutionary-original":
                results = workflow.discover_all_equations_llmsr_evolutionary(
                    output_dir=args.output_dir,
                    problem_filter=args.problem_filter
                )
            elif args.mode == "evolutionary":
                results = workflow.discover_all_equations_evolutionary(
                    output_dir=args.output_dir,
                    problem_filter=args.problem_filter
                )
            else:
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

