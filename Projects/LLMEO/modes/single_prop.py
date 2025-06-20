"""
Single property optimization mode

A mode focused on optimizing a single specific property, suitable for precise control of the optimization target.
"""

import sys
import os
from typing import Any, Dict

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from sci_demo.workflow import Workflow
from sci_demo.generation import Generation
from sci_demo.prompt import Prompt
from sci_demo.oracle import Oracle

from prompt import PROMPT_G
from _utils import (
    make_text_for_existing_tmcs,
    retrive_tmc_from_message,
    find_tmc_in_space,
)
import pandas as pd


def run_single_prop(args) -> Dict[str, Any]:
    """
    Run single property optimization mode

    Args:
        args: Command line argument object

    Returns:
        Workflow execution result
    """
    print("🎯 Run single property optimization mode...")

    # Setup components
    generator = Generation(
        openai_api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
    )

    # Load data
    with open("data/1M-space_50-ligands-full.csv", "r") as fo:
        df_ligands_str = fo.read()

    df_tmc = pd.read_csv("data/ground_truth_fitness_values.csv")
    df_ligands = pd.read_csv("data/1M-space_50-ligands-full.csv")
    LIG_CHARGE = {row["id"]: row["charge"] for _, row in df_ligands.iterrows()}

    # Prepare data
    tmc_samples = df_tmc.sample(n=args.samples, random_state=args.seed)
    text_tmc = make_text_for_existing_tmcs(tmc_samples, LIG_CHARGE, ["gap"])

    # Create prompt
    prompt = Prompt(
        custom_template=PROMPT_G,
        default_vars={
            "CSV_FILE_CONTENT": df_ligands_str,
            "CURRENT_SAMPLES": text_tmc,
            "NUM_PROVIDED_SAMPLES": len(tmc_samples),
            "NUM_SAMPLES": args.num_samples,
        },
    )
    print(prompt.build())
    # Setup evaluator
    oracle = Oracle()

    def improvement_rate_metric(
        history: dict, reference: any, current_iteration: int, **kwargs
    ) -> float:
        explored_tmc = history.setdefault("tmc_explorer", [tmc_samples])
        current_round_tmc = find_tmc_in_space(
            reference, retrive_tmc_from_message(history["outputs"][-1], 10)
        )
        if current_round_tmc is None or current_round_tmc.empty:
            print("No new TMC found")
            history["tmc_explorer"].append(pd.DataFrame())
        else:
            history["tmc_explorer"].append(current_round_tmc)
        all_tmc = pd.concat(history["tmc_explorer"])
        top10_avg_gap = all_tmc["gap"].nlargest(10).mean()
        return top10_avg_gap

    oracle.register_multi_round_metric("top10_avg_gap", improvement_rate_metric)

    def prompt_fn(iteration, history):
        if iteration > 1:
            new_tmc = retrive_tmc_from_message(history["outputs"][-1], 10)
            new_tmc_df = find_tmc_in_space(df_tmc, new_tmc)
            if new_tmc_df is None or new_tmc_df.empty:
                return prompt
            new_tmc_text = make_text_for_existing_tmcs(new_tmc_df, LIG_CHARGE, ["gap"])
            updated_current_samples = (
                prompt.default_vars["CURRENT_SAMPLES"] + new_tmc_text
            )
            prompt.add_vars(CURRENT_SAMPLES=updated_current_samples)
        return prompt

    workflow = Workflow(
        generator=generator,
        oracle=oracle,
        max_iterations=3,
        enable_multi_round_metrics=True,
    )

    result = workflow.run_sync(
        prompt=prompt_fn,
        reference=df_tmc,
        gen_args={"max_tokens": args.max_tokens, "temperature": args.temperature},
    )

    print(f"✅ Single property optimization completed! Score: {result['final_scores']}")
    return result


if __name__ == "__main__":
    # Test mode
    import argparse

    parser = argparse.ArgumentParser(
        description="Single property optimization mode test"
    )
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--samples", type=int, default=10, help="Initial sample number")
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Generated sample number"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=5000, help="Maximum token number"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature parameter"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    result = run_single_prop(args)
    print(f"Test completed: {result}")
