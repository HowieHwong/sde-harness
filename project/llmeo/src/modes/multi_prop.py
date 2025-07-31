"""
Multi-property optimization mode

A mode that iteratively optimizes multiple properties, supporting dynamic prompts and historical learning.
"""

import sys
import os
from typing import Any, Dict

import weave

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from sde_harness.core import Workflow, Generation, Prompt, Oracle

from ..utils.prompt import PROMPT_MB
from ..utils._utils import (
    make_text_for_existing_tmcs,
    retrive_tmc_from_message,
    find_tmc_in_space,
)
import pandas as pd


def run_multi_prop(args) -> Dict[str, Any]:
    """
    Run multi-property optimization mode

    Args:
        args: Command line argument object

    Returns:
        Workflow execution result
    """
    print("🔄 Run multi-property optimization mode...")

    # Setup components
    weave.init("LLMEO-multi_prop_mode")
    generator = Generation(models_file=project_root + "/models.yaml", credentials_file=project_root + "/credentials.yaml")
    # Load data
    with open("data/1M-space_50-ligands-full.csv", "r") as fo:
        df_ligands_str = fo.read()

    df_tmc = pd.read_csv("data/ground_truth_fitness_values.csv")
    df_ligands = pd.read_csv("data/1M-space_50-ligands-full.csv")
    LIG_CHARGE = {row["id"]: row["charge"] for _, row in df_ligands.iterrows()}

    # Prepare data
    tmc_samples = df_tmc.sample(n=args.samples, random_state=args.seed)
    text_tmc = make_text_for_existing_tmcs(
        tmc_samples, LIG_CHARGE, ["gap", "polarisability"]
    )

    # Create prompt
    prompt = Prompt(
        custom_template=PROMPT_MB,
        default_vars={
            "CSV_FILE_CONTENT": df_ligands_str,
            "CURRENT_SAMPLES": text_tmc,
            "NUM_PROVIDED_SAMPLES": len(tmc_samples),
            "NUM_SAMPLES": args.num_samples,
        },
    )
    print(prompt.build())

    # Setup evaluator - multi-round mode
    oracle = Oracle()

    def improvement_rate_metric(
        history: dict, reference: any, current_iteration: int, **kwargs
    ) -> float:
        # Initialize tmc_explorer with serializable format
        if "tmc_explorer" not in history:
            # Convert initial tmc_samples to serializable format
            history["tmc_explorer"] = [tmc_samples.to_dict('records')]
        
        current_round_tmc = find_tmc_in_space(
            reference, retrive_tmc_from_message(history["outputs"][-1], 10)
        )
        if current_round_tmc is None or current_round_tmc.empty:
            history["tmc_explorer"].append([])
        else:
            # Convert DataFrame to serializable format
            history["tmc_explorer"].append(current_round_tmc.to_dict('records'))
        
        # Convert back to DataFrame for calculations
        all_tmc_records = []
        for tmc_list in history["tmc_explorer"]:
            if tmc_list:  # Only add non-empty lists
                all_tmc_records.extend(tmc_list)
        
        if not all_tmc_records:
            return 0.0
        
        all_tmc = pd.DataFrame(all_tmc_records)
        all_tmc["score"] = all_tmc["gap"] * all_tmc["polarisability"]
        top10_avg_score = all_tmc["score"].nlargest(10).mean()
        return top10_avg_score

    oracle.register_multi_round_metric("top10_avg_score", improvement_rate_metric)

    # Dynamic prompt function
    def prompt_fn(iteration, history):
        if iteration > 1:
            new_tmc = retrive_tmc_from_message(history["outputs"][-1], 10)
            new_tmc_df = find_tmc_in_space(df_tmc, new_tmc)
            if new_tmc_df is None or new_tmc_df.empty:
                return prompt
            new_tmc_text = make_text_for_existing_tmcs(new_tmc_df, LIG_CHARGE, ["gap", "polarisability"])
            updated_current_samples = (
                prompt.default_vars["CURRENT_SAMPLES"] + new_tmc_text
            )
            prompt.add_vars(CURRENT_SAMPLES=updated_current_samples)
        return prompt

    # Create workflow
    workflow = Workflow(
        generator=generator,
        oracle=oracle,
        max_iterations=args.iterations,
        enable_multi_round_metrics=True,
    )

    # Run
    result = workflow.run_sync(
        prompt=prompt_fn,
        reference=df_tmc,
        gen_args={"model_name":args.model, "max_tokens": args.max_tokens, "temperature": args.temperature},
    )

    print(
        f"✅ Multi-property optimization completed! Final score: {result['final_scores']}"
    )
    return result