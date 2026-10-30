"""Command-line interface for LLM-SR equation discovery (symbolic regression)."""

import argparse
import os
import random
import sys
from typing import Any, Dict

import numpy as np

# Ensure sde_harness is importable by adding the repository root to sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sde_harness.base import CLIBase
from src.modes.evol import LLMSREvolutionaryWorkflow


class LLMSRCLI(CLIBase):
    """CLI for LLM-SR equation discovery. Defaults reproduce the paper runs."""

    def __init__(self):
        super().__init__("llm-sym-reg")

    def _add_project_arguments(self, parser: argparse.ArgumentParser):
        """Add LLM-SR specific arguments."""
        parser.add_argument(
            "--dataset",
            default="lsrtransform",
            choices=["lsrtransform", "bio_pop_growth", "chem_react", "matsci", "phys_osc"],
            help="LLM-SRBench dataset to use",
        )
        parser.add_argument("--problem", type=str, help="Single problem name to solve (e.g. PO0)")
        parser.add_argument("--problem-filter", nargs="+", help="List of problem names to process")
        parser.add_argument(
            "--model",
            default="openai/gpt-4o-2024-08-06",
            help="Model key in models.yaml (e.g. openai/gpt-5, openai/gpt-5-chat, "
                 "anthropic/claude-sonnet-4.5, deepseek/deepseek-R1)",
        )
        parser.add_argument("--max-samples", type=int, default=1000,
                            help="Total LLM samples per problem (default: 1000)")
        parser.add_argument("--samples-per-prompt", type=int, default=8,
                            help="LLM samples drawn per prompt (default: 8)")
        parser.add_argument("--num-islands", type=int, default=10,
                            help="Number of islands in the experience buffer (default: 10)")
        parser.add_argument("--functions-per-prompt", type=int, default=2,
                            help="Previous equation versions included in each prompt (default: 2)")
        parser.add_argument("--reset-period", type=int, default=4 * 60 * 60,
                            help="Seconds between resets of the weaker islands (default: 14400)")
        parser.add_argument("--cluster-temperature-init", type=float, default=0.1,
                            help="Initial cluster sampling temperature (default: 0.1)")
        parser.add_argument("--cluster-temperature-period", type=int, default=30_000,
                            help="Cluster sampling temperature period (default: 30000)")
        parser.add_argument("--evaluate-timeout", type=int, default=30,
                            help="Timeout (s) for evaluating one sampled equation (default: 30)")
        parser.add_argument("--hf-descriptions", action="store_true",
                            help="Use HuggingFace variable descriptions instead of the paper's")
        parser.add_argument("--seed", type=int, default=None, help="Random seed for the search")
        parser.add_argument("--weave", action="store_true", help="Enable Weave logging")
        parser.add_argument("--project-name", default="llm-sym-reg",
                            help="Weave project name for logging")

    def run_command(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Run LLM-SR."""
        print("Starting LLM-SR equation discovery")
        print(f"Dataset: {args.dataset} | Model: {args.model} | Samples: {args.max_samples} "
              f"({args.samples_per_prompt}/prompt) | Islands: {args.num_islands}")
        print(f"Output directory: {args.output_dir}")

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
        if args.weave:
            import weave
            weave.init(args.project_name)

        workflow = LLMSREvolutionaryWorkflow(
            model_name=args.model,
            models_file=args.models_file,
            credentials_file=args.credentials_file,
            max_samples=args.max_samples,
            samples_per_prompt=args.samples_per_prompt,
            num_islands=args.num_islands,
            functions_per_prompt=args.functions_per_prompt,
            reset_period=args.reset_period,
            cluster_sampling_temperature_init=args.cluster_temperature_init,
            cluster_sampling_temperature_period=args.cluster_temperature_period,
            evaluate_timeout_seconds=args.evaluate_timeout,
            use_paper_descriptions=not args.hf_descriptions,
        )
        workflow.setup_dataset(args.dataset)

        if args.problem:
            results = {args.problem: workflow.discover_equation(args.problem, args.output_dir)}
        else:
            results = workflow.discover_all_equations(
                output_dir=args.output_dir, problem_filter=args.problem_filter)

        workflow.print_summary()
        return {"results": results, "output_dir": args.output_dir}


def main():
    """Main entry point."""
    cli = LLMSRCLI()
    cli.main()


if __name__ == "__main__":
    main()
