import sys
import os
from typing import Any, Dict

import weave
from TMC import ten_tmc

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
tmc_samples = df_tmc.sample(n=10, random_state=42)
text_tmc = make_text_for_existing_tmcs(tmc_samples, LIG_CHARGE, ["gap"])

# Create prompt
prompt = Prompt(
    custom_template=PROMPT_G,
    default_vars={
        "CSV_FILE_CONTENT": df_ligands_str,
        "CURRENT_SAMPLES": text_tmc,
        "NUM_PROVIDED_SAMPLES": len(tmc_samples),
        "NUM_SAMPLES": 10,
    },
)
gen_arg = { "temperature": 1, "response_format": ten_tmc}
result = generator.generate(prompt=prompt.build(), model_name="anthropic/claude-3-7-sonnet-20250219", **gen_arg)
try:
    import json
    from TMC import ten_tmc
    tmc_dict = json.loads(result['text'])
    tmc_obj = ten_tmc(**tmc_dict)
    print(tmc_obj)
except:
    print(f"Error parsing response: {result}")

tmc_list = [i.ligands for i in tmc_obj.tmc]
print(len(tmc_list))
lig_df = find_tmc_in_space2(df_tmc, tmc_list)
print(lig_df)



