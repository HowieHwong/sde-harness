"""Tests for data loader functionality."""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import json
import tempfile
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import (
    load_dataset, 
    format_gene_list, 
    validate_data_files,
    read_task_prompt,
    load_essential_genes
)


class TestDataLoader(unittest.TestCase):
    """Test cases for data loader functions."""
    
    def test_format_gene_list(self):
        """Test gene list formatting."""
        # Test with empty list
        self.assertEqual(format_gene_list([]), "")
        
        # Test with single gene
        self.assertEqual(format_gene_list(["GENE1"]), "GENE1")
        
        # Test with multiple genes
        self.assertEqual(
            format_gene_list(["GENE1", "GENE2", "GENE3"]), 
            "GENE1, GENE2, GENE3"
        )
    
    def test_read_task_prompt(self):
        """Test reading task prompt from JSON."""
        # Create temporary JSON file with correct format
        test_data = {
            "Task": "Test task description",
            "Measurement": "Test measurement"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            # Test reading
            task, measurement = read_task_prompt(temp_path)
            
            # Verify
            self.assertEqual(task, "Test task description")
            self.assertEqual(measurement, "Test measurement")
        finally:
            # Clean up
            os.unlink(temp_path)
    
    @patch('numpy.load')
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch('builtins.open', new_callable=mock_open, read_data='{"Task": "Test", "Measurement": "Metric"}')
    def test_load_dataset(self, mock_file, mock_read_csv, mock_exists, mock_np_load):
        """Test loading dataset."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Create mock dataframes for train and test
        mock_train = pd.DataFrame({
            'GENE1': [0.5, 0.3, 0.2],
            'GENE2': [-0.3, -0.1, 0.4],
            'GENE3': [0.8, 0.6, 0.7]
        })
        mock_test = pd.DataFrame({
            'GENE1': [0.4],
            'GENE2': [-0.2],
            'GENE3': [0.9]
        })
        
        # Set up read_csv to return different dataframes
        mock_read_csv.side_effect = [mock_train, mock_test]
        
        # Mock numpy load for topmovers
        mock_np_load.return_value = np.array(['GENE1', 'GENE2'])
        
        # Test loading
        result = load_dataset("test_dataset")
        
        # Verify structure based on actual implementation
        self.assertIn('ground_truth', result)  # The function returns ground_truth, not train/test
        self.assertIn('task_description', result)
        self.assertIn('measurement', result)
        self.assertIn('topmovers', result)
        
        # Verify data
        self.assertEqual(result['task_description'], 'Test')
        self.assertEqual(result['measurement'], 'Metric')
        self.assertEqual(result['ground_truth'].shape[1], 3)  # 3 genes
        self.assertEqual(len(result['topmovers']), 2)
    
    @patch('os.path.exists')
    def test_validate_data_files(self, mock_exists):
        """Test data file validation."""
        # Test when all files exist
        mock_exists.return_value = True
        result = validate_data_files("test_dataset")
        self.assertTrue(result)
        
        # The function actually checks for topmovers file
        mock_exists.assert_called_with('datasets/topmovers_test_dataset.npy')
        
        # Test when files don't exist
        mock_exists.return_value = False
        result = validate_data_files("nonexistent_dataset")
        self.assertFalse(result)
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_load_essential_genes(self, mock_read_csv, mock_exists):
        """Test loading essential genes."""
        # Test when file exists
        mock_exists.return_value = True
        mock_df = pd.DataFrame({
            'GENE': ['ESSENTIAL1', 'ESSENTIAL2', 'ESSENTIAL3']
        })
        mock_read_csv.return_value = mock_df
        
        genes = load_essential_genes("test_path.txt")
        
        # Verify
        self.assertEqual(len(genes), 3)
        self.assertIn('ESSENTIAL1', genes)
        mock_read_csv.assert_called_once_with("test_path.txt", delimiter='\t')
        
        # Test when file doesn't exist
        mock_exists.return_value = False
        genes = load_essential_genes("nonexistent.txt")
        self.assertEqual(genes, [])


if __name__ == '__main__':
    unittest.main()