"""
Few-Shot learning mode

A mode based on few samples for learning, suitable for rapid exploration and experiments.
"""

import sys
import os
from typing import Any, Dict

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from sci_demo.workflow import Workflow
from sci_demo.generation import Generation
from sci_demo.prompt import Prompt
from sci_demo.oracle import Oracle

from prompt import PROMPT_G
from _utils import make_text_for_existing_tmcs, retrive_tmc_from_message, find_tmc_in_space
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
            "NUM_SAMPLES": args.num_samples
        }
    )
    
    # Setup evaluator
    oracle = Oracle()
    def top10_avg_gap(response: str, reference: Any) -> float:
        current_round_tmc = find_tmc_in_space(reference, retrive_tmc_from_message(response, 10))
        if current_round_tmc is None or current_round_tmc.empty:
            return 0
        else:
            return current_round_tmc["gap"].nlargest(10).mean()
    
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
        gen_args={
            "max_tokens": args.max_tokens, 
            "temperature": args.temperature
        }
    )
    
    print(f"✅ Few-Shot mode completed! Final score: {result['final_scores']}")
    return result


if __name__ == "__main__":
    # Test mode
    import argparse
    
    parser = argparse.ArgumentParser(description="Few-Shot learning mode test")
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    parser.add_argument('--samples', type=int, default=10, help='Initial sample number')
    parser.add_argument('--num-samples', type=int, default=10, help='Generated sample number')
    parser.add_argument('--max-tokens', type=int, default=5000, help='Maximum token number')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature parameter')
    parser.add_argument('--iterations', type=int, default=2, help='Iteration number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    result = run_few_shot(args)
    print(f"Test completed: {result}") 