import sys
import os
from typing import Any, Dict, List

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sci_demo.workflow import Workflow
from sci_demo.generation import Generation
from sci_demo.prompt import Prompt
from sci_demo.oracle import Oracle

from prompt import PROMPT_G
import pandas as pd
from _utils import make_text_for_existing_tmcs, retrive_tmc_from_message, find_tmc_in_space

generator = Generation(
     openai_api_key="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",  
)

with open("data/1M-space_50-ligands-full.csv", "r") as fo:
    df_ligands_str = fo.read()

df_tmc = pd.read_csv("data/ground_truth_fitness_values.csv")
df_ligands = pd.read_csv("data/1M-space_50-ligands-full.csv")
LIG_CHARGE = {row["id"]: row["charge"] for _, row in df_ligands.iterrows()}
tmc_samples = df_tmc.sample(n=10, random_state=42)
text_tmc = make_text_for_existing_tmcs(tmc_samples, LIG_CHARGE, ["gap"])

prompt = Prompt(
    custom_template=PROMPT_G,
    default_vars={"CSV_FILE_CONTENT": df_ligands_str,
                  "CURRENT_SAMPLES": text_tmc,
                  "NUM_PROVIDED_SAMPLES": len(tmc_samples),
                  "NUM_SAMPLES": 10}
)

def improvement_rate_metric(history: Dict[str, List[Any]], reference: Any, current_iteration: int, **kwargs) -> float:
    # print("history: "+str(history))
    explored_tmc = history.setdefault("tmc_explorer", [tmc_samples])
    current_round_tmc = find_tmc_in_space(reference, retrive_tmc_from_message(history["outputs"][-1], 10))
    if(current_round_tmc is None or current_round_tmc.empty):
        history["tmc_explorer"].append(pd.DataFrame())
    else:
        history["tmc_explorer"].append(current_round_tmc)
    all_tmc = pd.concat(history["tmc_explorer"])
    # print("all_tmc: "+str(all_tmc))
    top10_avg_gap = all_tmc["gap"].nlargest(10).mean()
    # print("top10_avg_gap: "+str(top10_avg_gap))
    return top10_avg_gap
    
oracle = Oracle()
oracle.register_multi_round_metric("top10_avg_gap", improvement_rate_metric)


# Dynamic prompt function
def prompt_fn(iteration, history):
    if(iteration > 1):
        new_tmc = retrive_tmc_from_message(history["outputs"][-1], 10)
        new_tmc_df = find_tmc_in_space(df_tmc, new_tmc)
        if(new_tmc_df is None or new_tmc_df.empty):
            return prompt
        new_tmc_text = make_text_for_existing_tmcs(new_tmc_df, LIG_CHARGE, ["gap"])
        updated_current_samples = prompt.default_vars["CURRENT_SAMPLES"] + new_tmc_text
        prompt.add_vars(CURRENT_SAMPLES=updated_current_samples)
    return prompt

workflow = Workflow(generator=generator, oracle=oracle, max_iterations=2, enable_multi_round_metrics=True)

result = workflow.run_sync(
    prompt= prompt_fn,
    reference= df_tmc,
    gen_args={"max_tokens": 5000, "temperature": 0.0}
)




