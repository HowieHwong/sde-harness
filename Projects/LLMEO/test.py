from typing import Any, Dict, List
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os
import weave


# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from _utils import (
    make_text_for_existing_tmcs,
    retrive_tmc_from_message,
    find_tmc_in_space,
)
from prompt import PROMPT_G
from sci_demo.workflow import Workflow
from sci_demo.generation import Generation
from sci_demo.prompt import Prompt
from sci_demo.oracle import Oracle


class TestUtils(unittest.TestCase):
    """Test utils.py"""

    def test_Generation(self):
        """Test Generation class"""
        weave.init("LLMEO-test")
        gen = Generation()
        self.assertIsNotNone(gen)
    
    def test_model(self):
        """Test model"""
        weave.init("LLMEO-test")
        gen = Generation()

        # response = gen.generate("Hello, nice to meet you", model_name="openai/gpt-4o-2024-08-06")
        # self.assertIsNotNone(response)
        # print("GPT-4o: Worked")
        # response = gen.generate("Hello, nice to meet you", model_name="deepseek/deepseek-chat")
        # self.assertIsNotNone(response)
        # print("DeepSeek-Chat: Worked")
        # response = gen.generate("Hello, nice to meet you", model_name="anthropic/claude-3-7-sonnet-20250219")
        # self.assertIsNotNone(response)
        # print("Claude-3.7-Sonnet: Worked")
        response = gen.generate("Hello, nice to meet you", model_name="openai/o3")
        self.assertIsNotNone(response)
        print("GPT-O3: Worked")
        print(response.message.content)


    def test_dataset_exists(self):
        """Test if dataset exists"""
        self.assertTrue(os.path.exists("data/1M-space_50-ligands-full.csv"))
        self.assertTrue(os.path.exists("data/ground_truth_fitness_values.csv"))

    def test_prompt(self):
        with open("data/1M-space_50-ligands-full.csv", "r") as fo:
            df_ligands_str = fo.read()

        df_tmc = pd.read_csv("data/ground_truth_fitness_values.csv")
        df_ligands = pd.read_csv("data/1M-space_50-ligands-full.csv")
        LIG_CHARGE = {row["id"]: row["charge"] for _, row in df_ligands.iterrows()}
        tmc_samples = df_tmc.sample(n=10, random_state=42)
        text_tmc = make_text_for_existing_tmcs(tmc_samples, LIG_CHARGE, ["gap"])
        self.assertEqual(len(tmc_samples), 10)
        self.assertIsNotNone(text_tmc)
        prompt = Prompt(
            custom_template=PROMPT_G,
            default_vars={
                "CSV_FILE_CONTENT": df_ligands_str,
                "CURRENT_SAMPLES": text_tmc,
                "NUM_PROVIDED_SAMPLES": len(tmc_samples),
                "NUM_SAMPLES": 10,
            },
        )
        self.assertIsNotNone(prompt)

    def test_oracle(self):
        def improvement_rate_metric(
            history: Dict[str, List[Any]],
            reference: Any,
            current_iteration: int,
            **kwargs
        ) -> float:
            return 1.0

        oracle = Oracle()
        oracle.register_multi_round_metric("top10_avg_gap", improvement_rate_metric)
        self.assertEqual(oracle.list_multi_round_metrics(), ["top10_avg_gap"])


if __name__ == "__main__":
    # unittest.main()
    txt = '''
 the HOMO-LUMO gap in Pd(II) square planar complexes, we aim for a combination of strong-field ligands (e.g., π-acceptors like nitriles, carbonyls, or aromatic N-donors) and charge balance. The provided data shows that TMCs with neutral total charge and ligands like [C-]#[N+] (BICRIQ-subgraph-3), N-donors (KEYRUB-subgraph-1), and O-donors (IJIMIX-subgraph-1) tend to have larger gaps. We also note that negatively charged ligands (e.g., [O-], [N-]) can stabilize the metal center when paired with neutral or weakly donating ligands. Here, we propose new TMCs by:
1. Combining strong π-acceptors (e.g., BICRIQ-subgraph-3, CIGDAA-subgraph-1) with neutral N-donors (e.g., RUCBEY-subgraph-1, KULGAZ-subgraph-2) to enhance ligand field splitting.
2. Using negatively charged ligands (e.g., [O-] from ZOQFIU-subgraph-0) with neutral ligands to balance charge and maintain a high gap.
3. Avoiding bulky or weakly coordinating ligands (e.g., halides) unless paired with strong-field ligands to compensate.

<<<TMC>>>: 
[
    Pd_BICRIQ-subgraph-3_RUCBEY-subgraph-1_IJIMIX-subgraph-1_KULGAZ-subgraph-2,
    Pd_CIGDAA-subgraph-1_RUCBEY-subgraph-1_ZOQFIU-subgraph-0_BOSJIF-subgraph-1,
    Pd_BICRIQ-subgraph-3_ZOQFIU-subgraph-0_NURKEQ-subgraph-2_CEVJAP-subgraph-2,
    Pd_CIGDAA-subgraph-1_NURKEQ-subgraph-2_IJIMIX-subgraph-1_KULGAZ-subgraph-2,
    Pd_BICRIQ-subgraph-3_ZOQFIU-subgraph-0_RUCBEY-subgraph-1_CEVJAP-subgraph-2,
    Pd_CIGDAA-subgraph-1_KULGAZ-subgraph-2_IJIMIX-subgraph-1_RUCBEY-subgraph-1,
    Pd_BICRIQ-subgraph-3_IJIMIX-subgraph-1_NURKEQ-subgraph-2_KULGAZ-subgraph-2,
    Pd_CIGDAA-subgraph-1_ZOQFIU-subgraph-0_BICRIQ-subgraph-3_CEVJAP-subgraph-2,
    Pd_BICRIQ-subgraph-3_RUCBEY-subgraph-1_ZOQFIU-subgraph-0_KULGAZ-subgraph-2,
    Pd_CIGDAA-subgraph-1_NURKEQ-subgraph-2_ZOQFIU-subgraph-0_CEVJAP-subgraph-2
]

<<<TOTAL_CHARGE>>>: 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

<<<gap>>>: 
[3.2, 3.0, 3.1, 3.3, 3.4, 3.2, 3.5, 3.6, 3.3, 3.4]

**Rationale for Predictions**:
- **BICRIQ-subgraph-3** ([C-]#[N+]) is a strong π-acceptor, enhancing the ligand field.
- **CIGDAA-subgraph-1** ([C-]#[O+]) is another strong π-acceptor, similar to BICRIQ.
- **ZOQFIU-subgraph-0** ([O-]c1ccccc1) provides charge balance and 
'''
    new_tmc = retrive_tmc_from_message(txt, 10)
    print(new_tmc)
