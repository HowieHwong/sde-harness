"""Tests for bio metrics evaluator."""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluators.bio_metrics import (
    BioEvaluator, 
    calculate_hit_rate,
    calculate_precision_recall,
    evaluate_round,
    aggregate_results
)


class TestBioMetrics(unittest.TestCase):
    """Test cases for bio metrics functions."""
    
    def test_calculate_hit_rate(self):
        """Test hit rate calculation."""
        # Test with empty predicted genes
        hit_rate = calculate_hit_rate([], ['GENE1', 'GENE2'])
        self.assertEqual(hit_rate, 0.0)
        
        # Test with all hits (genes in ground truth)
        hit_rate = calculate_hit_rate(['GENE1', 'GENE2'], ['GENE1', 'GENE2', 'GENE3'])
        self.assertEqual(hit_rate, 1.0)
        
        # Test with some hits
        hit_rate = calculate_hit_rate(['GENE1', 'GENE2', 'NONEXISTENT'], ['GENE1', 'GENE2', 'GENE3'])
        self.assertAlmostEqual(hit_rate, 2/3)
        
        # Test with no hits
        hit_rate = calculate_hit_rate(['NONEXISTENT1', 'NONEXISTENT2'], ['GENE1', 'GENE2'])
        self.assertEqual(hit_rate, 0.0)
        
        # Test with essential genes exclusion
        hit_rate = calculate_hit_rate(
            ['GENE1', 'GENE2', 'ESSENTIAL1'], 
            ['GENE1', 'GENE2', 'ESSENTIAL1'],
            essential_genes=['ESSENTIAL1']
        )
        self.assertEqual(hit_rate, 1.0)  # ESSENTIAL1 excluded from both sets
    
    def test_calculate_precision_recall(self):
        """Test precision and recall calculation."""
        # Test basic case
        result = calculate_precision_recall(['GENE1', 'GENE2', 'GENE3'], ['GENE1', 'GENE2', 'GENE4'])
        
        self.assertIn('precision', result)
        self.assertIn('recall', result)
        self.assertIn('f1_score', result)
        
        # Precision: 2/3 (GENE1, GENE2 are hits out of 3 predicted)
        self.assertAlmostEqual(result['precision'], 2/3)
        # Recall: 2/3 (GENE1, GENE2 found out of 3 ground truth)
        self.assertAlmostEqual(result['recall'], 2/3)
    
    def test_evaluate_round(self):
        """Test round evaluation."""
        result = evaluate_round(
            ['GENE1', 'GENE2', 'GENE3'],
            ['GENE1', 'GENE2', 'GENE4'],
            round_num=1
        )
        
        # Check all required fields
        self.assertIn('round', result)
        self.assertIn('num_predicted', result)
        self.assertIn('hit_rate', result)
        self.assertIn('precision', result)
        self.assertIn('recall', result)
        self.assertIn('f1_score', result)
        self.assertIn('predicted_genes', result)
        
        self.assertEqual(result['round'], 1)
        self.assertEqual(result['num_predicted'], 3)
    
    def test_aggregate_results(self):
        """Test results aggregation."""
        results = [
            {'hit_rate': 0.7, 'precision': 0.8, 'recall': 0.6, 'f1_score': 0.7, 'predicted_genes': ['GENE1', 'GENE2']},
            {'hit_rate': 0.8, 'precision': 0.85, 'recall': 0.75, 'f1_score': 0.8, 'predicted_genes': ['GENE2', 'GENE3']},
            {'hit_rate': 0.75, 'precision': 0.82, 'recall': 0.7, 'f1_score': 0.75, 'predicted_genes': ['GENE1', 'GENE4']}
        ]
        
        agg = aggregate_results(results)
        
        # Check structure
        self.assertIn('mean_hit_rate', agg)
        self.assertIn('mean_precision', agg)
        self.assertIn('mean_recall', agg)
        self.assertIn('mean_f1_score', agg)
        self.assertIn('total_unique_genes', agg)
        
        # Check values
        self.assertAlmostEqual(agg['mean_hit_rate'], 0.75)
        self.assertAlmostEqual(agg['mean_precision'], 0.823, places=2)
        self.assertEqual(agg['total_unique_genes'], 4)  # GENE1, GENE2, GENE3, GENE4


class TestBioEvaluator(unittest.TestCase):
    """Test cases for BioEvaluator class."""
    
    @patch('os.path.exists')
    @patch('numpy.load')
    @patch('pandas.read_csv')
    def test_bio_evaluator_init(self, mock_read_csv, mock_np_load, mock_exists):
        """Test BioEvaluator initialization."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock essential genes
        mock_essential_df = pd.DataFrame({'GENE': ['ESSENTIAL1', 'ESSENTIAL2']})
        mock_read_csv.return_value = mock_essential_df
        
        # Mock ground truth
        mock_np_load.return_value = np.array(['GENE1', 'GENE2', 'GENE3'])
        
        # Initialize evaluator
        evaluator = BioEvaluator('test_dataset')
        
        # Verify
        self.assertEqual(evaluator.dataset_name, 'test_dataset')
        self.assertEqual(len(evaluator.essential_genes), 2)
        self.assertEqual(len(evaluator.ground_truth), 3)
    
    @patch('os.path.exists')
    @patch('numpy.load')
    @patch('pandas.read_csv')
    def test_bio_evaluator_evaluate(self, mock_read_csv, mock_np_load, mock_exists):
        """Test BioEvaluator evaluate method."""
        # Setup mocks
        mock_exists.return_value = True
        mock_essential_df = pd.DataFrame({'GENE': ['ESSENTIAL1']})
        mock_read_csv.return_value = mock_essential_df
        mock_np_load.return_value = np.array(['GENE1', 'GENE2', 'GENE3'])
        
        # Initialize and evaluate
        evaluator = BioEvaluator('test_dataset')
        result = evaluator.evaluate(['GENE1', 'GENE2', 'NONEXISTENT'])
        
        # Verify structure
        self.assertIn('round', result)
        self.assertIn('num_predicted', result)
        self.assertIn('hit_rate', result)
        self.assertIn('predicted_genes', result)
        
        # Verify values
        self.assertEqual(result['num_predicted'], 3)
        self.assertAlmostEqual(result['hit_rate'], 2/3)
    
    @patch('os.path.exists')
    @patch('numpy.load')
    @patch('pandas.read_csv')
    def test_bio_evaluator_get_hits(self, mock_read_csv, mock_np_load, mock_exists):
        """Test getting hit genes."""
        # Setup mocks
        mock_exists.return_value = True
        mock_essential_df = pd.DataFrame({'GENE': []})
        mock_read_csv.return_value = mock_essential_df
        mock_np_load.return_value = np.array(['GENE1', 'GENE2', 'GENE3'])
        
        # Initialize and get hits
        evaluator = BioEvaluator('test_dataset')
        hits = evaluator.get_hits(['GENE1', 'GENE3', 'NONEXISTENT'])
        
        # Verify
        self.assertEqual(sorted(hits), ['GENE1', 'GENE3'])
    
    @patch('os.path.exists')
    @patch('numpy.load')
    def test_horlbeck_dataset_special_case(self, mock_np_load, mock_exists):
        """Test Horlbeck dataset special handling."""
        # Mock file existence
        mock_exists.side_effect = lambda path: 'topmovers_Horlbeck.npy' in path
        
        # Mock ground truth as pairs
        mock_np_load.return_value = np.array([['GENE1', 'GENE2'], ['GENE3', 'GENE4']])
        
        # Initialize evaluator
        evaluator = BioEvaluator('Horlbeck')
        
        # Verify gene pairs are formatted correctly
        self.assertEqual(evaluator.ground_truth, ['GENE1_GENE2', 'GENE3_GENE4'])


if __name__ == '__main__':
    unittest.main()