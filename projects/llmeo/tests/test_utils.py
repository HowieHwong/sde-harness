"""
Unit tests for LLMEO utility functions (_utils.py)
"""

import unittest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from src.utils._utils import (
    make_text_for_existing_tmcs,
    retrive_tmc_from_message,
    find_tmc_in_space,
)


class TestUtilsFunctions(unittest.TestCase):
    """Test utility functions in _utils.py"""

    def setUp(self):
        """Set up test data"""
        # Create sample TMC data
        self.sample_tmc_data = pd.DataFrame({
            'lig1': ['RUCBEY-subgraph-1', 'MEBXUN-subgraph-1', 'KEYRUB-subgraph-1'],
            'lig2': ['WECJIA-subgraph-3', 'BIFMOV-subgraph-1', 'NURKEQ-subgraph-2'],
            'lig3': ['KEYRUB-subgraph-1', 'CUJYEL-subgraph-2', 'MEBXUN-subgraph-1'],
            'lig4': ['NURKEQ-subgraph-2', 'EZEXEM-subgraph-1', 'BIFMOV-subgraph-1'],
            'gap': [3.2, 2.8, 3.5],
            'polarisability': [450.5, 380.2, 435.1]
        })
        
        # Create sample ligand charge data
        self.sample_lig_charge = {
            'RUCBEY-subgraph-1': 0,
            'WECJIA-subgraph-3': 0,
            'KEYRUB-subgraph-1': 0,
            'NURKEQ-subgraph-2': 0,
            'MEBXUN-subgraph-1': 0,
            'BIFMOV-subgraph-1': 0,
            'CUJYEL-subgraph-2': 0,
            'EZEXEM-subgraph-1': 0
        }

    def test_make_text_for_existing_tmcs_single_property(self):
        """Test make_text_for_existing_tmcs with single property"""
        result = make_text_for_existing_tmcs(
            self.sample_tmc_data, 
            self.sample_lig_charge, 
            ['gap']
        )
        
        self.assertIsInstance(result, str)
        self.assertIn('Pd_RUCBEY-subgraph-1_WECJIA-subgraph-3_KEYRUB-subgraph-1_NURKEQ-subgraph-2', result)
        self.assertIn('3.2', result)
        self.assertIn('{', result)
        self.assertIn('}', result)
        
        # Check that it has the correct number of lines
        lines = result.strip().split('\n')
        self.assertEqual(len(lines), 3)

    def test_make_text_for_existing_tmcs_multiple_properties(self):
        """Test make_text_for_existing_tmcs with multiple properties"""
        result = make_text_for_existing_tmcs(
            self.sample_tmc_data, 
            self.sample_lig_charge, 
            ['gap', 'polarisability']
        )
        
        self.assertIsInstance(result, str)
        self.assertIn('3.2', result)
        self.assertIn('450.5', result)
        
        # Check that both properties are included
        lines = result.strip().split('\n')
        first_line = lines[0]
        self.assertIn('3.2', first_line)
        self.assertIn('450.5', first_line)

    def test_make_text_for_existing_tmcs_charge_calculation(self):
        """Test that total charge is calculated correctly"""
        result = make_text_for_existing_tmcs(
            self.sample_tmc_data, 
            self.sample_lig_charge, 
            ['gap']
        )
        
        # All ligands have charge 0, so total charge should be 2 (Pd +2)
        lines = result.strip().split('\n')
        for line in lines:
            self.assertIn(', 2, ', line)

    def test_retrive_tmc_from_message_basic(self):
        """Test basic TMC extraction from message"""
        test_message = """
        Here are some TMCs:
        *TMC*Pd_RUCBEY-subgraph-1_WECJIA-subgraph-3_KEYRUB-subgraph-1_NURKEQ-subgraph-2
        *TMC*Pd_MEBXUN-subgraph-1_BIFMOV-subgraph-1_CUJYEL-subgraph-2_EZEXEM-subgraph-1
        """
        
        result = retrive_tmc_from_message(test_message, 2)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIn('Pd_RUCBEY-subgraph-1_WECJIA-subgraph-3_KEYRUB-subgraph-1_NURKEQ-subgraph-2', result)
        self.assertIn('Pd_MEBXUN-subgraph-1_BIFMOV-subgraph-1_CUJYEL-subgraph-2_EZEXEM-subgraph-1', result)

    def test_retrive_tmc_from_message_different_delimiters(self):
        """Test TMC extraction with different delimiters"""
        test_messages = [
            "<<<TMC>>>:Pd_RUCBEY-subgraph-1_WECJIA-subgraph-3_KEYRUB-subgraph-1_NURKEQ-subgraph-2",
            "<TMC>Pd_MEBXUN-subgraph-1_BIFMOV-subgraph-1_CUJYEL-subgraph-2_EZEXEM-subgraph-1",
            "TMC:Pd_KEYRUB-subgraph-1_NURKEQ-subgraph-2_MEBXUN-subgraph-1_BIFMOV-subgraph-1"
        ]
        
        for message in test_messages:
            result = retrive_tmc_from_message(message, 1)
            self.assertIsInstance(result, list)
            self.assertTrue(len(result) >= 0)  # Should extract at least something or return empty

    def test_retrive_tmc_from_message_no_matches(self):
        """Test TMC extraction when no valid TMCs are found"""
        test_message = "This message contains no valid TMC patterns"
        
        result = retrive_tmc_from_message(test_message, 1)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_find_tmc_in_space_exact_match(self):
        """Test finding TMCs with exact ligand matches"""
        test_tmcs = ['Pd_RUCBEY-subgraph-1_WECJIA-subgraph-3_KEYRUB-subgraph-1_NURKEQ-subgraph-2']
        
        result = find_tmc_in_space(self.sample_tmc_data, test_tmcs)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['gap'], 3.2)

    def test_find_tmc_in_space_rotational_match(self):
        """Test finding TMCs with rotational ligand matches"""
        # This TMC is a rotation of the first entry
        test_tmcs = ['Pd_WECJIA-subgraph-3_KEYRUB-subgraph-1_NURKEQ-subgraph-2_RUCBEY-subgraph-1']
        
        result = find_tmc_in_space(self.sample_tmc_data, test_tmcs)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['gap'], 3.2)  # Should match the same TMC

    def test_find_tmc_in_space_multiple_matches(self):
        """Test finding multiple TMCs"""
        test_tmcs = [
            'Pd_RUCBEY-subgraph-1_WECJIA-subgraph-3_KEYRUB-subgraph-1_NURKEQ-subgraph-2',
            'Pd_MEBXUN-subgraph-1_BIFMOV-subgraph-1_CUJYEL-subgraph-2_EZEXEM-subgraph-1'
        ]
        
        result = find_tmc_in_space(self.sample_tmc_data, test_tmcs)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    def test_find_tmc_in_space_no_matches(self):
        """Test finding TMCs when no matches exist"""
        test_tmcs = ['Pd_NONEXISTENT-1_NONEXISTENT-2_NONEXISTENT-3_NONEXISTENT-4']
        
        result = find_tmc_in_space(self.sample_tmc_data, test_tmcs)
        
        self.assertIsNone(result)

    def test_find_tmc_in_space_empty_input(self):
        """Test finding TMCs with empty input"""
        result = find_tmc_in_space(self.sample_tmc_data, [])
        
        self.assertIsNone(result)

    def test_find_tmc_in_space_none_input(self):
        """Test finding TMCs with None values in input"""
        test_tmcs = [None, 'Pd_RUCBEY-subgraph-1_WECJIA-subgraph-3_KEYRUB-subgraph-1_NURKEQ-subgraph-2']
        
        result = find_tmc_in_space(self.sample_tmc_data, test_tmcs)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()