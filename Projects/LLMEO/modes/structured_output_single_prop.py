"""
Single property optimization mode

A mode focused on optimizing a single specific property, suitable for precise control of the optimization target.
"""

import json
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
    find_tmc_in_space2,
    make_text_for_existing_tmcs,
    retrive_tmc_from_message,
    find_tmc_in_space,
)
import pandas as pd
from pydantic import BaseModel
class TMC(BaseModel):
    center_metal: str
    ligands: list[str]
    expected_homolumo_gap: float

class ten_tmc(BaseModel):
    tmc: list[TMC]
    gap: list[float]

def run_single_prop_structured(args) -> Dict[str, Any]:
    """
    Run single property optimization mode

    Args:
        args: Command line argument object

    Returns:
        Workflow execution result
    """
    print("🎯 Run single property optimization mode...")
    weave.init("LLMEO-single-prop")
    # Setup components
    generator = Generation(models_file=project_root + "/models.yaml", credentials_file=project_root + "/credentials.yaml", max_workers=1)

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
    # print(prompt.build())
    # Setup evaluator
    oracle = Oracle()

    def improvement_rate_metric(
        history: dict, reference: any, current_iteration: int, **kwargs
    ) -> float:
        # Initialize tmc_explorer with serializable format
        if "tmc_explorer" not in history:
            # Convert initial tmc_samples to serializable format
            history["tmc_explorer"] = [tmc_samples.to_dict('records')]
        
        tmc_dict = json.loads(history["outputs"][-1])
        tmc_obj = ten_tmc(**tmc_dict)
        print(tmc_obj)

        current_round_tmc = find_tmc_in_space2(
            reference, [i.ligands for i in tmc_obj.tmc]
        )
        if current_round_tmc is None or current_round_tmc.empty:
            print("No new TMC found")
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
        top10_avg_gap = all_tmc["gap"].nlargest(10).mean()
        return top10_avg_gap

    oracle.register_multi_round_metric("top10_avg_gap", improvement_rate_metric)

    def prompt_fn(iteration, history):
        if iteration > 1:
            print("=========================================")
            # print(history["outputs"][-1])
            tmc_dict = json.loads(history["outputs"][-1])
            tmc_obj = ten_tmc(**tmc_dict)
            print(tmc_obj)
            new_tmc_df = find_tmc_in_space2(df_tmc, [i.ligands for i in tmc_obj.tmc])
            print("=========================================")
            print(new_tmc_df)
            if new_tmc_df is None or new_tmc_df.empty:
                print("No new TMC found")
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
        max_iterations=args.iterations,
        enable_multi_round_metrics=True,
    )

    result = workflow.run_sync(
        prompt=prompt_fn,
        reference=df_tmc,
        gen_args={"model_name": args.model, "max_tokens": args.max_tokens, "temperature": args.temperature, "response_format": ten_tmc},
    )

    print(f"✅ Single property optimization completed! Score: {result['final_scores']}")
    return result
