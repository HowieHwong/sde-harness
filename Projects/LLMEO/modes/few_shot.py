"""
Few-Shot learning mode

A mode based on few samples for learning, suitable for rapid exploration and experiments.
"""

import sys
import os
from typing import Any, Dict

import weave

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


def run_few_shot(args) -> Dict[str, Any]:
    """
    Run few-shot learning mode

    Args:
        args: Command line argument object

    Returns:
        Workflow execution result
    """
    print("🚀 Run Few-Shot learning mode...")
    weave.init("few_shot_mode")
    # Setup components
    generator = Generation(models_file=project_root + "/models.yaml", credentials_file=project_root + "/credentials.yaml")

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

    # Setup evaluator
    oracle = Oracle()

    def top10_avg_gap(response: str, reference: Any) -> float:
        current_round_tmc = find_tmc_in_space(
            reference, retrive_tmc_from_message(response, 10)
        )
        if current_round_tmc is None or current_round_tmc.empty or len(current_round_tmc) == 0:
            return 0
        else:
            try:
                return current_round_tmc["gap"].nlargest(10).mean()
            except:
                return 0

    oracle.register_metric("top10_avg_gap", top10_avg_gap)

    # Create workflow
    workflow = Workflow(
        generator=generator,
        oracle=oracle,
        max_iterations=args.iterations,
        enable_multi_round_metrics=False
    )

    # Run
    result = workflow.run_sync(
        prompt=prompt,
        reference=df_tmc,
        gen_args={"model_name":args.model, "temperature": args.temperature, "max_tokens": args.max_tokens},
    )

    print(f"✅ Few-Shot mode completed! Final score: {result['final_scores']}")
    return result
