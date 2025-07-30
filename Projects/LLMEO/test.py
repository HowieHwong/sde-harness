"""
Legacy test file for LLMEO - maintained for backward compatibility

For comprehensive testing, use the tests/ directory with pytest:
    python -m pytest tests/
    
Or use the test runner:
    python tests/run_tests.py
"""

from typing import Any, Dict, List
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os
import weave


# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from src.utils._utils import (
    make_text_for_existing_tmcs,
    retrive_tmc_from_message,
    find_tmc_in_space,
)
from src.utils.prompt import PROMPT_G
from sde_harness.core import Workflow, Generation, Prompt, Oracle


class TestUtils(unittest.TestCase):
    """Legacy test class - for comprehensive tests see tests/ directory"""

    def test_Generation(self):
        """Test Generation class"""
        weave.init("LLMEO-test")
        gen = Generation(models_file=project_root + "/models.yaml", credentials_file=project_root + "/credentials.yaml")
        self.assertIsNotNone(gen)
    
    def test_model(self):
        """Test model"""
        weave.init("LLMEO-test")
        gen = Generation(models_file=project_root + "/models.yaml", credentials_file=project_root + "/credentials.yaml")

        # response = gen.generate("Hello, nice to meet you", model_name="openai/gpt-4o-2024-08-06")
        # self.assertIsNotNone(response)
        # print("GPT-4o: Worked")
        # response = gen.generate("Hello, nice to meet you", model_name="deepseek/deepseek-chat")
        # self.assertIsNotNone(response)
        # print("DeepSeek-Chat: Worked")
        # response = gen.generate("Hello, nice to meet you", model_name="anthropic/claude-3-7-sonnet-20250219")
        # self.assertIsNotNone(response)
        # print("Claude-3.7-Sonnet: Worked")
        # Test with a basic model (commented out to avoid API calls in CI)
        # response = gen.generate("Hello, nice to meet you", model_name="openai/gpt-4o-mini")
        # self.assertIsNotNone(response)
        # print("GPT-4o-mini: Worked")
        # print(response.message.content if hasattr(response, 'message') else response)
        
        # Just verify the generator was created successfully
        print("Generation instance created successfully")


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
    print("Running legacy tests...")
    print("For comprehensive testing, use: python -m pytest tests/")
    print("Or: python tests/run_tests.py")
    print()
    unittest.main()
