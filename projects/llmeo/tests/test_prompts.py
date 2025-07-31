"""
Unit tests for LLMEO prompt templates (prompt.py)
"""

import unittest
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from src.utils.prompt import PROMPT_G, PROMPT_MB
from sde_harness.core import Prompt


class TestPromptTemplates(unittest.TestCase):
    """Test prompt templates and their functionality"""

    def setUp(self):
        """Set up test data"""
        self.sample_csv_content = """SMILES,id,charge,connecting atom element,connecting atom index
c1ccccn1,RUCBEY-subgraph-1,0,N,1
CP(C)C,WECJIA-subgraph-3,0,P,1"""
        
        self.sample_current_samples = """{Pd_RUCBEY-subgraph-1_WECJIA-subgraph-3_KEYRUB-subgraph-1_NURKEQ-subgraph-2, 2, 3.2}
{Pd_MEBXUN-subgraph-1_BIFMOV-subgraph-1_CUJYEL-subgraph-2_EZEXEM-subgraph-1, 2, 2.8}"""
        
        self.sample_vars = {
            "CSV_FILE_CONTENT": self.sample_csv_content,
            "CURRENT_SAMPLES": self.sample_current_samples,
            "NUM_PROVIDED_SAMPLES": 2,
            "NUM_SAMPLES": 5
        }

    def test_prompt_g_structure(self):
        """Test that PROMPT_G has the correct structure and placeholders"""
        self.assertIsInstance(PROMPT_G, str)
        
        # Check for required placeholders
        required_placeholders = [
            "{CSV_FILE_CONTENT}",
            "{CURRENT_SAMPLES}",
            "{NUM_PROVIDED_SAMPLES}",
            "{NUM_SAMPLES}"
        ]
        
        for placeholder in required_placeholders:
            self.assertIn(placeholder, PROMPT_G)
        
        # Check for key content
        key_content = [
            "pool of 50 ligands",
            "Pd based square planer transition metal complex",
            "HOMO-LUMO gap",
            "total charge of the TMC to be -1, 0 or 1"
        ]
        
        for content in key_content:
            self.assertIn(content, PROMPT_G)

    def test_prompt_mb_structure(self):
        """Test that PROMPT_MB has the correct structure and placeholders"""
        self.assertIsInstance(PROMPT_MB, str)
        
        # Check for required placeholders
        required_placeholders = [
            "{CSV_FILE_CONTENT}",
            "{CURRENT_SAMPLES}",
            "{NUM_PROVIDED_SAMPLES}",
            "{NUM_SAMPLES}"
        ]
        
        for placeholder in required_placeholders:
            self.assertIn(placeholder, PROMPT_MB)
        
        # Check for multi-property specific content
        multi_prop_content = [
            "polarisability",
            "HOMO-LUMO gap (> 4 eV) and polarisability (> 400 au)",
            "simultaneously maximize both HOMO-LUMO gap"
        ]
        
        for content in multi_prop_content:
            self.assertIn(content, PROMPT_MB)

    def test_prompt_g_with_sde_harness_prompt_class(self):
        """Test PROMPT_G integration with SDE harness Prompt class"""
        prompt = Prompt(
            custom_template=PROMPT_G,
            default_vars=self.sample_vars
        )
        
        self.assertIsNotNone(prompt)
        
        # Build the prompt and check content
        built_prompt = prompt.build()
        
        # Check that variables are properly substituted
        self.assertIn(self.sample_csv_content, built_prompt)
        self.assertIn(self.sample_current_samples, built_prompt)
        self.assertIn("2", built_prompt)  # NUM_PROVIDED_SAMPLES
        self.assertIn("5", built_prompt)  # NUM_SAMPLES
        
        # Check that placeholders are replaced
        self.assertNotIn("{CSV_FILE_CONTENT}", built_prompt)
        self.assertNotIn("{CURRENT_SAMPLES}", built_prompt)

    def test_prompt_mb_with_sde_harness_prompt_class(self):
        """Test PROMPT_MB integration with SDE harness Prompt class"""
        prompt = Prompt(
            custom_template=PROMPT_MB,
            default_vars=self.sample_vars
        )
        
        self.assertIsNotNone(prompt)
        
        # Build the prompt and check content
        built_prompt = prompt.build()
        
        # Check that variables are properly substituted
        self.assertIn(self.sample_csv_content, built_prompt)
        self.assertIn(self.sample_current_samples, built_prompt)
        
        # Check for multi-property specific content in built prompt
        self.assertIn("polarisability", built_prompt)

    def test_prompt_g_output_format_requirements(self):
        """Test that PROMPT_G specifies correct output format"""
        output_format_elements = [
            "<<<Explaination>>>:",
            "<<<TMC>>>:",
            "<<<TOTAL_CHARGE>>>:",
            "<<<gap>>>:"
        ]
        
        for element in output_format_elements:
            self.assertIn(element, PROMPT_G)

    def test_prompt_mb_output_format_requirements(self):
        """Test that PROMPT_MB specifies correct output format"""
        output_format_elements = [
            "<<<Explaination>>>:",
            "<<<TMC>>>:",
            "<<<TOTAL_CHARGE>>>:",
            "<<<gap>>>:",
            "<<<polarisability>>>:"
        ]
        
        for element in output_format_elements:
            self.assertIn(element, PROMPT_MB)

    def test_prompt_template_consistency(self):
        """Test consistency between PROMPT_G and PROMPT_MB"""
        # Both should have common requirements
        common_requirements = [
            "pool of 50 ligands",
            "Pd_$L1_$L2_$L3_$L4",
            "total charge of the TMC to be -1, 0 or 1",
            "DO NOT propose duplicated TMCs"
        ]
        
        for requirement in common_requirements:
            self.assertIn(requirement, PROMPT_G)
            self.assertIn(requirement, PROMPT_MB)

    def test_prompt_variable_substitution_edge_cases(self):
        """Test prompt with edge case variable values"""
        edge_case_vars = {
            "CSV_FILE_CONTENT": "",
            "CURRENT_SAMPLES": "",
            "NUM_PROVIDED_SAMPLES": 0,
            "NUM_SAMPLES": 1
        }
        
        prompt_g = Prompt(
            custom_template=PROMPT_G,
            default_vars=edge_case_vars
        )
        
        prompt_mb = Prompt(
            custom_template=PROMPT_MB,
            default_vars=edge_case_vars
        )
        
        # Should not raise exceptions
        built_g = prompt_g.build()
        built_mb = prompt_mb.build()
        
        self.assertIsInstance(built_g, str)
        self.assertIsInstance(built_mb, str)
        
        # Check that numeric values are properly converted
        self.assertIn("0", built_g)
        self.assertIn("1", built_g)

    def test_prompt_instructions_completeness(self):
        """Test that prompts contain comprehensive instructions"""
        instruction_elements = [
            "chemistry knowledge",
            "ground-truth",
            "ligand crossover",
            "ligand mutations",
            "clockwise ordering",
            "cyclic symmetry"
        ]
        
        for element in instruction_elements:
            self.assertIn(element, PROMPT_G)
            self.assertIn(element, PROMPT_MB)

    def test_prompt_constraints(self):
        """Test that prompts specify necessary constraints"""
        constraints = [
            "All ligands in the TMC need to be those present in this csv file",
            "control the total charge of the TMC to be -1, 0, or 1",
            "DO NOT propose duplicated TMCs"
        ]
        
        for constraint in constraints:
            self.assertIn(constraint, PROMPT_G)
            self.assertIn(constraint, PROMPT_MB)


if __name__ == '__main__':
    unittest.main()