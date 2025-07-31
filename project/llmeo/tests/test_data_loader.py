"""
Unit tests for LLMEO data loading functions (data_loader.py)
"""

import unittest
import pandas as pd
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from src.utils.data_loader import (
    load_data,
    setup_generator,
    create_prompt,
    setup_oracle,
    validate_data_files
)
from src.utils.prompt import PROMPT_G, PROMPT_MB


class TestDataLoaderFunctions(unittest.TestCase):
    """Test data loading functions in data_loader.py"""

    def setUp(self):
        """Set up test data"""
        # Sample ligands CSV content
        self.sample_ligands_csv = """SMILES,id,charge,connecting atom element,connecting atom index
c1ccccn1,RUCBEY-subgraph-1,0,N,1
CP(C)C,WECJIA-subgraph-3,0,P,1
N#CC,KEYRUB-subgraph-1,0,N,1
[C-]#[N+]c1c(C)cccc1C,NURKEQ-subgraph-2,0,C,1"""

        # Sample fitness CSV content
        self.sample_fitness_csv = """lig1,lig2,lig3,lig4,gap,polarisability
RUCBEY-subgraph-1,WECJIA-subgraph-3,KEYRUB-subgraph-1,NURKEQ-subgraph-2,3.2,450.5
WECJIA-subgraph-3,KEYRUB-subgraph-1,NURKEQ-subgraph-2,RUCBEY-subgraph-1,2.8,380.2"""

        # Sample TMC samples DataFrame
        self.sample_tmc_samples = pd.DataFrame({
            'lig1': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'lig2': ['WECJIA-subgraph-3', 'KEYRUB-subgraph-1'],
            'lig3': ['KEYRUB-subgraph-1', 'NURKEQ-subgraph-2'],
            'lig4': ['NURKEQ-subgraph-2', 'RUCBEY-subgraph-1'],
            'gap': [3.2, 2.8],
            'polarisability': [450.5, 380.2]
        })

    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_load_data_success(self, mock_exists, mock_read_csv, mock_file_open):
        """Test successful data loading"""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock file content
        mock_file_open.return_value.read.return_value = self.sample_ligands_csv
        
        # Mock pandas read_csv
        mock_ligands_df = pd.DataFrame({
            'id': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'charge': [0, 0]
        })
        mock_fitness_df = pd.DataFrame({
            'lig1': ['RUCBEY-subgraph-1'],
            'gap': [3.2]
        })
        
        mock_read_csv.side_effect = [mock_fitness_df, mock_ligands_df]
        
        # Test the function
        df_ligands_str, df_tmc, df_ligands, lig_charge = load_data()
        
        # Assertions
        self.assertEqual(df_ligands_str, self.sample_ligands_csv)
        self.assertIsInstance(df_tmc, pd.DataFrame)
        self.assertIsInstance(df_ligands, pd.DataFrame)
        self.assertIsInstance(lig_charge, dict)
        self.assertEqual(lig_charge['RUCBEY-subgraph-1'], 0)

    @patch('os.path.exists')
    def test_load_data_file_not_found_ligands(self, mock_exists):
        """Test load_data when ligands file doesn't exist"""
        mock_exists.side_effect = lambda x: x != "data/1M-space_50-ligands-full.csv"
        
        with self.assertRaises(FileNotFoundError) as context:
            load_data()
        
        self.assertIn("Ligand file not found", str(context.exception))

    @patch('os.path.exists')
    def test_load_data_file_not_found_fitness(self, mock_exists):
        """Test load_data when fitness file doesn't exist"""
        mock_exists.side_effect = lambda x: x != "data/ground_truth_fitness_values.csv"
        
        with self.assertRaises(FileNotFoundError) as context:
            load_data()
        
        self.assertIn("Fitness file not found", str(context.exception))

    def test_setup_generator_default_params(self):
        """Test setup_generator with default parameters"""
        generator = setup_generator()
        
        # Check that it returns a Generation object
        self.assertIsNotNone(generator)
        # We can't test much more without actual files, but structure is correct

    def test_setup_generator_custom_params(self):
        """Test setup_generator with custom parameters"""
        generator = setup_generator("custom_models.yaml", "custom_credentials.yaml")
        
        self.assertIsNotNone(generator)

    def test_create_prompt_basic(self):
        """Test basic prompt creation"""
        df_ligands_str = "sample,ligands,data"
        text_tmc = "sample tmc text"
        num_samples = 10
        num_provided_samples = 5
        
        prompt = create_prompt(df_ligands_str, text_tmc, num_samples, num_provided_samples)
        
        self.assertIsNotNone(prompt)
        # Check that the prompt has the expected variables
        built_prompt = prompt.build()
        self.assertIn("sample,ligands,data", built_prompt)
        self.assertIn("sample tmc text", built_prompt)
        self.assertIn("10", built_prompt)
        self.assertIn("5", built_prompt)

    def test_setup_oracle_single_round(self):
        """Test oracle setup for single round mode"""
        df_tmc = self.sample_tmc_samples
        tmc_samples = self.sample_tmc_samples
        
        oracle = setup_oracle(df_tmc, tmc_samples, mode="single_round")
        
        self.assertIsNotNone(oracle)
        # Check that the metric is registered
        self.assertIn("top10_avg_gap", oracle.list_metrics())

    def test_setup_oracle_multi_round(self):
        """Test oracle setup for multi round mode"""
        df_tmc = self.sample_tmc_samples
        tmc_samples = self.sample_tmc_samples
        
        oracle = setup_oracle(df_tmc, tmc_samples, mode="multi_round")
        
        self.assertIsNotNone(oracle)
        # Check that the multi-round metric is registered
        self.assertIn("top10_avg_gap", oracle.list_multi_round_metrics())

    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_validate_data_files_success(self, mock_exists, mock_read_csv):
        """Test successful data file validation"""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock pandas DataFrames with required columns
        mock_ligands_df = pd.DataFrame({
            'id': ['test-1', 'test-2'],
            'charge': [0, -1],
            'SMILES': ['CC', 'CN']
        })
        mock_fitness_df = pd.DataFrame({
            'lig1': ['test-1'],
            'lig2': ['test-2'],
            'lig3': ['test-1'],
            'lig4': ['test-2'],
            'gap': [3.2]
        })
        
        mock_read_csv.side_effect = [mock_ligands_df, mock_fitness_df]
        
        result = validate_data_files()
        
        self.assertTrue(result)

    @patch('os.path.exists')
    def test_validate_data_files_missing_ligands(self, mock_exists):
        """Test validation when ligands file is missing"""
        mock_exists.side_effect = lambda x: x != "data/1M-space_50-ligands-full.csv"
        
        result = validate_data_files()
        
        self.assertFalse(result)

    @patch('os.path.exists')
    def test_validate_data_files_missing_fitness(self, mock_exists):
        """Test validation when fitness file is missing"""
        mock_exists.side_effect = lambda x: x != "data/ground_truth_fitness_values.csv"
        
        result = validate_data_files()
        
        self.assertFalse(result)

    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_validate_data_files_missing_columns_ligands(self, mock_exists, mock_read_csv):
        """Test validation when ligands file is missing required columns"""
        mock_exists.return_value = True
        
        # Mock DataFrame missing 'charge' column
        mock_ligands_df = pd.DataFrame({
            'id': ['test-1', 'test-2'],
            'SMILES': ['CC', 'CN']
        })
        mock_fitness_df = pd.DataFrame({
            'lig1': ['test-1'],
            'lig2': ['test-2'],
            'lig3': ['test-1'],
            'lig4': ['test-2'],
            'gap': [3.2]
        })
        
        mock_read_csv.side_effect = [mock_ligands_df, mock_fitness_df]
        
        result = validate_data_files()
        
        self.assertFalse(result)

    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_validate_data_files_missing_columns_fitness(self, mock_exists, mock_read_csv):
        """Test validation when fitness file is missing required columns"""
        mock_exists.return_value = True
        
        # Mock DataFrame missing 'gap' column
        mock_ligands_df = pd.DataFrame({
            'id': ['test-1', 'test-2'],
            'charge': [0, -1]
        })
        mock_fitness_df = pd.DataFrame({
            'lig1': ['test-1'],
            'lig2': ['test-2'],
            'lig3': ['test-1'],
            'lig4': ['test-2']
        })
        
        mock_read_csv.side_effect = [mock_ligands_df, mock_fitness_df]
        
        result = validate_data_files()
        
        self.assertFalse(result)

    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_validate_data_files_exception_handling(self, mock_exists, mock_read_csv):
        """Test validation handles exceptions gracefully"""
        mock_exists.return_value = True
        mock_read_csv.side_effect = Exception("Test exception")
        
        result = validate_data_files()
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()