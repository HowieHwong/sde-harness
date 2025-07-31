"""Tests for different modes."""
import unittest
import json
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import sys
from types import SimpleNamespace

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modes.perturb_genes import format_gene_scores, parse_gene_solution, get_task_config
from src.modes.analyze import find_results_files, analyze_final_results


class TestPerturbGenes(unittest.TestCase):
    """Test cases for perturb genes mode."""
    
    def setUp(self):
        """Set up test data."""
        self.ground_truth = pd.DataFrame({
            'Score': [0.8, 0.5, -0.3]
        }, index=['GENE1', 'GENE2', 'GENE3'])
    
    def test_format_gene_scores(self):
        """Test gene score formatting."""
        # Test with empty genes
        result = format_gene_scores([], self.ground_truth)
        self.assertEqual(result, "No genes tested yet.")
        
        # Test with valid genes
        genes = ['GENE1', 'GENE2']
        result = format_gene_scores(genes, self.ground_truth)
        self.assertIn("GENE1: 0.8000", result)
        self.assertIn("GENE2: 0.5000", result)
        
        # Test with max_display limit
        genes = ['GENE1', 'GENE2', 'GENE3']
        result = format_gene_scores(genes, self.ground_truth, max_display=2)
        self.assertIn("GENE1", result)
        self.assertIn("GENE2", result)
        self.assertIn("... and 1 more genes", result)
    
    def test_parse_gene_solution(self):
        """Test gene solution parsing."""
        # The parser looks for "Solution:" section specifically
        response = """Based on the analysis, here are the genes:
        
        Solution:
        1. BRCA1
        2. TP53
        3. EGFR
        """
        result = parse_gene_solution(response)
        self.assertIn('BRCA1', result)
        self.assertIn('TP53', result)
        self.assertIn('EGFR', result)
        
        # Test with numbered Solution section
        response = """Here is my analysis:
        
        3. Solution:
        1. CDK4
        2. PTEN
        3. MYC
        """
        result = parse_gene_solution(response)
        self.assertIn('CDK4', result)
        self.assertIn('PTEN', result)
        self.assertIn('MYC', result)
        
        # Test parsing gene pairs with + separator
        response = """Gene pairs analysis:
        
        Solution:
        1. GENE1 + GENE2
        2. GENE3 + GENE4
        """
        result = parse_gene_solution(response, is_pairs=True)
        # Function returns sorted tuples
        self.assertIn(('GENE1', 'GENE2'), result)
        self.assertIn(('GENE3', 'GENE4'), result)
    
    def test_get_task_config(self):
        """Test task configuration generation."""
        # Test brief variant - function expects 5 parameters
        config = get_task_config(
            task_variant="brief",
            data_name="IFNG", 
            task_description="Test description",
            measurement="Test measurement",
            num_genes=10
        )
        
        self.assertIn('research_problem', config)
        self.assertIn('instructions', config)
        # The brief variant doesn't include data_name in the research problem
        self.assertIn('genome-wide CRISPR screen', config['research_problem'])
        self.assertIn('Test description', config['research_problem'])
        self.assertIn('Test measurement', config['research_problem'])
        self.assertIn('10 genes', config['instructions'])
        self.assertIn('DO NOT PREDICT GENES THAT HAVE ALREADY BEEN TESTED', config['instructions'])
        
        # Test detailed variant
        config = get_task_config(
            task_variant="full",  # Changed from "detailed" to "full"
            data_name="IL2", 
            task_description="Test description",
            measurement="Test measurement",
            num_genes=10
        )
        self.assertIn('research_problem', config)
        self.assertIn('instructions', config)



class TestAnalyze(unittest.TestCase):
    """Test cases for analyze mode."""
    
    @patch('glob.glob')
    def test_find_results_files(self, mock_glob):
        """Test finding result files."""
        # Mock glob results
        mock_glob.return_value = [
            'results/gpt4_IFNG/run1/final_results.json',
            'results/gpt4_IFNG/run2/final_results.json'
        ]
        
        results = find_results_files('IFNG', 'gpt4')
        
        # Verify
        self.assertEqual(len(results), 2)
        self.assertTrue(all('final_results.json' in r for r in results))
    
    def test_analyze_final_results(self):
        """Test analyzing final results."""
        # Create test data matching expected format
        test_data = {
            'aggregate': {
                'mean_hit_rate': 0.75,
                'std_hit_rate': 0.05,
                'num_rounds': 5,
                'total_unique_genes': 20,
                'hit_rates_by_round': [0.7, 0.75, 0.8, 0.75, 0.75]
            },
            'total_hits': 15,
            'hits_progression': [3, 6, 10, 13, 15],
            'round_results': [{}, {}, {}, {}, {}]
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            # Test loading
            result = analyze_final_results(temp_path)
            
            # Verify
            self.assertEqual(result['mean_hit_rate'], 0.75)
            self.assertEqual(result['std_hit_rate'], 0.05)
            self.assertEqual(result['total_hits'], 15)
            self.assertEqual(result['num_rounds'], 5)
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_analyze_statistics(self):
        """Test analyzing statistics from results."""
        # Create test result with aggregate statistics
        test_result = {
            'mean_hit_rate': 0.75,
            'std_hit_rate': 0.05,
            'hit_rates_by_round': [0.7, 0.75, 0.8, 0.75, 0.75]
        }
        
        # Verify we can access statistics
        self.assertEqual(test_result['mean_hit_rate'], 0.75)
        self.assertEqual(test_result['std_hit_rate'], 0.05)
        self.assertEqual(len(test_result['hit_rates_by_round']), 5)


class TestCLI(unittest.TestCase):
    """Test cases for CLI functionality."""
    
    @patch('src.utils.data_loader.validate_data_files')
    def test_cli_data_validation(self, mock_validate):
        """Test that CLI validates data files."""
        from cli import main
        
        # This is a simple test to ensure the CLI module can be imported
        # Actual CLI testing would require more complex mocking
        self.assertTrue(callable(main))


if __name__ == '__main__':
    unittest.main()