"""
Unit tests for sde_harness.core.oracle module
"""

import unittest
import sys
import os
from unittest.mock import MagicMock
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sde_harness.core.oracle import Oracle


class TestOracle(unittest.TestCase):
    """Test Oracle class functionality"""

    def setUp(self):
        """Set up test data"""
        self.oracle = Oracle()
        
        # Sample metric functions
        def accuracy_metric(prediction, reference, **kwargs):
            return 0.85
        
        def bleu_metric(prediction, reference, **kwargs):
            return 0.72
        
        def multi_round_metric(history, reference, current_iteration, **kwargs):
            return current_iteration * 0.1
        
        self.sample_metrics = {
            'accuracy': accuracy_metric,
            'bleu': bleu_metric
        }
        self.multi_round_metric = multi_round_metric

    def test_oracle_init_empty(self):
        """Test Oracle initialization with no metrics"""
        oracle = Oracle()
        
        self.assertIsInstance(oracle.metrics, dict)
        self.assertIsInstance(oracle.multi_round_metrics, dict)
        self.assertEqual(len(oracle.metrics), 0)
        self.assertEqual(len(oracle.multi_round_metrics), 0)

    def test_oracle_init_with_metrics(self):
        """Test Oracle initialization with predefined metrics"""
        oracle = Oracle(metrics=self.sample_metrics)
        
        self.assertEqual(len(oracle.metrics), 2)
        self.assertIn('accuracy', oracle.metrics)
        self.assertIn('bleu', oracle.metrics)

    def test_register_metric_success(self):
        """Test successful metric registration"""
        def new_metric(prediction, reference, **kwargs):
            return 0.9
        
        self.oracle.register_metric('new_metric', new_metric)
        
        self.assertIn('new_metric', self.oracle.metrics)
        self.assertEqual(self.oracle.metrics['new_metric'], new_metric)

    def test_register_metric_duplicate_name(self):
        """Test registering metric with duplicate name"""
        def metric1(prediction, reference, **kwargs):
            return 0.8
        
        def metric2(prediction, reference, **kwargs):
            return 0.9
        
        self.oracle.register_metric('test_metric', metric1)
        
        with self.assertRaises(ValueError) as context:
            self.oracle.register_metric('test_metric', metric2)
        
        self.assertIn("already registered", str(context.exception))

    def test_register_multi_round_metric_success(self):
        """Test successful multi-round metric registration"""
        self.oracle.register_multi_round_metric('multi_test', self.multi_round_metric)
        
        self.assertIn('multi_test', self.oracle.multi_round_metrics)
        self.assertEqual(self.oracle.multi_round_metrics['multi_test'], self.multi_round_metric)

    def test_register_multi_round_metric_duplicate_name(self):
        """Test registering multi-round metric with duplicate name"""
        def metric1(history, reference, current_iteration, **kwargs):
            return 0.1
        
        def metric2(history, reference, current_iteration, **kwargs):
            return 0.2
        
        self.oracle.register_multi_round_metric('multi_test', metric1)
        
        with self.assertRaises(ValueError) as context:
            self.oracle.register_multi_round_metric('multi_test', metric2)
        
        self.assertIn("already registered", str(context.exception))

    def test_unregister_metric_success(self):
        """Test successful metric unregistration"""
        self.oracle.register_metric('temp_metric', self.sample_metrics['accuracy'])
        
        self.assertIn('temp_metric', self.oracle.metrics)
        
        self.oracle.unregister_metric('temp_metric')
        
        self.assertNotIn('temp_metric', self.oracle.metrics)

    def test_unregister_multi_round_metric_success(self):
        """Test successful multi-round metric unregistration"""
        self.oracle.register_multi_round_metric('temp_multi', self.multi_round_metric)
        
        self.assertIn('temp_multi', self.oracle.multi_round_metrics)
        
        self.oracle.unregister_metric('temp_multi')
        
        self.assertNotIn('temp_multi', self.oracle.multi_round_metrics)

    def test_unregister_nonexistent_metric(self):
        """Test unregistering non-existent metric (should raise KeyError)"""
        # Should raise KeyError for non-existent metric
        with self.assertRaises(KeyError):
            self.oracle.unregister_metric('nonexistent_metric')

    def test_list_metrics(self):
        """Test listing all registered metrics (both single-round and multi-round)"""
        self.oracle.register_metric('accuracy', self.sample_metrics['accuracy'])
        self.oracle.register_metric('bleu', self.sample_metrics['bleu'])
        self.oracle.register_multi_round_metric('multi1', self.multi_round_metric)
        
        metrics = self.oracle.list_metrics()
        
        self.assertIsInstance(metrics, list)
        self.assertIn('accuracy', metrics)
        self.assertIn('bleu', metrics)
        self.assertIn('multi1', metrics)
        self.assertEqual(len(metrics), 3)

    def test_list_multi_round_metrics(self):
        """Test listing registered multi-round metrics"""
        self.oracle.register_multi_round_metric('multi1', self.multi_round_metric)
        self.oracle.register_multi_round_metric('multi2', self.multi_round_metric)
        
        metrics = self.oracle.list_multi_round_metrics()
        
        self.assertIsInstance(metrics, list)
        self.assertIn('multi1', metrics)
        self.assertIn('multi2', metrics)
        self.assertEqual(len(metrics), 2)

    def test_evaluate_single_metric(self):
        """Test evaluating with a single metric"""
        self.oracle.register_metric('accuracy', self.sample_metrics['accuracy'])
        
        result = self.oracle.compute("prediction", "reference", metrics=['accuracy'])
        
        self.assertIsInstance(result, dict)
        self.assertIn('accuracy', result)
        self.assertEqual(result['accuracy'], 0.85)

    def test_evaluate_multiple_metrics(self):
        """Test evaluating with multiple metrics"""
        self.oracle.register_metric('accuracy', self.sample_metrics['accuracy'])
        self.oracle.register_metric('bleu', self.sample_metrics['bleu'])
        
        result = self.oracle.compute("prediction", "reference", metrics=['accuracy', 'bleu'])
        
        self.assertIsInstance(result, dict)
        self.assertIn('accuracy', result)
        self.assertIn('bleu', result)
        self.assertEqual(result['accuracy'], 0.85)
        self.assertEqual(result['bleu'], 0.72)

    def test_evaluate_all_metrics(self):
        """Test evaluating with all registered metrics"""
        self.oracle.register_metric('accuracy', self.sample_metrics['accuracy'])
        self.oracle.register_metric('bleu', self.sample_metrics['bleu'])
        
        result = self.oracle.compute("prediction", "reference")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn('accuracy', result)
        self.assertIn('bleu', result)

    def test_evaluate_nonexistent_metric(self):
        """Test evaluating with non-existent metric"""
        with self.assertRaises(KeyError):
            self.oracle.compute("prediction", "reference", metrics=['nonexistent'])

    def test_evaluate_multi_round_single_metric(self):
        """Test evaluating multi-round metrics"""
        self.oracle.register_multi_round_metric('improvement', self.multi_round_metric)
        
        history = {'outputs': ['output1', 'output2'], 'scores': [0.1, 0.2]}
        result = self.oracle.compute_with_history("current_prediction", "reference", history, 2, metrics=['improvement'])
        
        self.assertIsInstance(result, dict)
        self.assertIn('improvement', result)
        self.assertEqual(result['improvement'], 0.2)  # 2 * 0.1

    def test_evaluate_multi_round_all_metrics(self):
        """Test evaluating all multi-round metrics"""
        self.oracle.register_multi_round_metric('improvement', self.multi_round_metric)
        
        history = {'outputs': ['output1'], 'scores': [0.1]}
        result = self.oracle.compute_with_history("current_prediction", "reference", history, 1)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertIn('improvement', result)

    def test_evaluate_with_kwargs(self):
        """Test evaluation with additional keyword arguments"""
        def metric_with_kwargs(prediction, reference, threshold=0.5, **kwargs):
            return threshold * 2
        
        self.oracle.register_metric('kwargs_metric', metric_with_kwargs)
        
        result = self.oracle.compute("prediction", "reference", 
                                    metrics=['kwargs_metric'], threshold=0.8)
        
        self.assertEqual(result['kwargs_metric'], 1.6)  # 0.8 * 2

    def test_metric_function_signature_validation(self):
        """Test that metrics have proper function signatures"""
        # This is more of a documentation test - metrics should follow the expected signature
        def valid_metric(prediction, reference, **kwargs):
            return 1.0
        
        def valid_multi_round_metric(history, reference, current_iteration, **kwargs):
            return 1.0
        
        # These should not raise exceptions
        self.oracle.register_metric('valid', valid_metric)
        self.oracle.register_multi_round_metric('valid_multi', valid_multi_round_metric)
        
        # Test that they can be called
        result = self.oracle.compute("pred", "ref", metrics=['valid'])
        multi_result = self.oracle.compute_with_history("current_pred", "ref", {}, 1, metrics=['valid_multi'])
        
        self.assertEqual(result['valid'], 1.0)
        self.assertEqual(multi_result['valid_multi'], 1.0)

    def test_oracle_state_management(self):
        """Test that Oracle maintains its state correctly"""
        # Register some metrics
        self.oracle.register_metric('metric1', self.sample_metrics['accuracy'])
        self.oracle.register_multi_round_metric('multi1', self.multi_round_metric)
        
        # Check initial state
        self.assertEqual(len(self.oracle.list_metrics()), 2)  # All metrics (single + multi)
        self.assertEqual(len(self.oracle.list_single_round_metrics()), 1)  # Single-round only
        self.assertEqual(len(self.oracle.list_multi_round_metrics()), 1)  # Multi-round only
        
        # Add more metrics
        self.oracle.register_metric('metric2', self.sample_metrics['bleu'])
        
        # Check updated state
        self.assertEqual(len(self.oracle.list_metrics()), 3)  # All metrics (single + multi)
        self.assertEqual(len(self.oracle.list_single_round_metrics()), 2)  # Single-round only
        self.assertEqual(len(self.oracle.list_multi_round_metrics()), 1)  # Multi-round only
        
        # Remove metrics
        self.oracle.unregister_metric('metric1')
        
        # Check final state
        self.assertEqual(len(self.oracle.list_metrics()), 2)  # All metrics (single + multi)
        self.assertEqual(len(self.oracle.list_single_round_metrics()), 1)  # Single-round only
        self.assertNotIn('metric1', self.oracle.list_metrics())
        self.assertIn('metric2', self.oracle.list_metrics())


class TestOracleAdvanced(unittest.TestCase):
    """Advanced tests for Oracle functionality"""

    def test_complex_multi_round_metric(self):
        """Test complex multi-round metric with history analysis"""
        def improvement_rate_metric(history, reference, current_iteration, **kwargs):
            """Calculate improvement rate based on score history"""
            if 'scores' not in history or len(history['scores']) < 2:
                return 0.0
            
            scores = history['scores']
            if len(scores) == 1:
                return scores[0]
            
            # Calculate improvement rate
            recent_scores = scores[-2:]
            improvement = recent_scores[-1] - recent_scores[-2]
            return max(0.0, improvement)
        
        oracle = Oracle()
        oracle.register_multi_round_metric('improvement_rate', improvement_rate_metric)
        
        # Test with improving scores
        history1 = {'scores': [0.5, 0.7, 0.8]}
        result1 = oracle.compute_with_history("current_pred", "ref", history1, 3, metrics=['improvement_rate'])
        self.assertAlmostEqual(result1['improvement_rate'], 0.1, places=5)  # 0.8 - 0.7
        
        # Test with declining scores
        history2 = {'scores': [0.8, 0.6, 0.5]}
        result2 = oracle.compute_with_history("current_pred", "ref", history2, 3, metrics=['improvement_rate'])
        self.assertEqual(result2['improvement_rate'], 0.0)  # max(0.0, 0.5 - 0.6)

    def test_metric_with_reference_data_analysis(self):
        """Test metric that analyzes reference data"""
        def reference_similarity_metric(prediction, reference, **kwargs):
            """Mock metric that analyzes reference data"""
            if isinstance(reference, list):
                return len(reference) * 0.1  # Mock calculation based on reference size
            return 0.5
        
        oracle = Oracle()
        oracle.register_metric('ref_similarity', reference_similarity_metric)
        
        # Test with list reference
        result1 = oracle.compute("prediction", [1, 2, 3, 4, 5], metrics=['ref_similarity'])
        self.assertEqual(result1['ref_similarity'], 0.5)  # 5 * 0.1
        
        # Test with non-list reference
        result2 = oracle.compute("prediction", "single_ref", metrics=['ref_similarity'])
        self.assertEqual(result2['ref_similarity'], 0.5)

    def test_oracle_error_handling(self):
        """Test Oracle error handling in metric execution"""
        def error_prone_metric(prediction, reference, **kwargs):
            if prediction == "error":
                raise ValueError("Test error")
            return 1.0
        
        oracle = Oracle()
        oracle.register_metric('error_metric', error_prone_metric)
        
        # Test normal execution
        result = oracle.compute("normal", "ref", metrics=['error_metric'])
        self.assertEqual(result['error_metric'], 1.0)
        
        # Test error handling - Oracle should propagate the error
        with self.assertRaises(ValueError):
            oracle.compute("error", "ref", metrics=['error_metric'])


@pytest.mark.integration
class TestOracleIntegration(unittest.TestCase):
    """Integration tests for Oracle class"""

    def test_oracle_with_real_world_metrics(self):
        """Test Oracle with realistic metric implementations"""
        def accuracy_metric(predictions, labels, **kwargs):
            """Calculate accuracy for classification"""
            if len(predictions) != len(labels):
                return 0.0
            
            correct = sum(1 for p, l in zip(predictions, labels) if p == l)
            return correct / len(predictions)
        
        def f1_score_metric(predictions, labels, **kwargs):
            """Mock F1 score calculation"""
            # Simplified F1 calculation for testing
            if not predictions or not labels:
                return 0.0
            
            # Mock precision and recall
            precision = 0.8
            recall = 0.7
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * (precision * recall) / (precision + recall)
        
        oracle = Oracle()
        oracle.register_metric('accuracy', accuracy_metric)
        oracle.register_metric('f1_score', f1_score_metric)
        
        # Test with sample data
        predictions = ['A', 'B', 'A', 'C', 'B']
        labels = ['A', 'B', 'C', 'C', 'B']
        
        results = oracle.compute(predictions, labels)
        
        self.assertIn('accuracy', results)
        self.assertIn('f1_score', results)
        self.assertEqual(results['accuracy'], 0.8)  # 4/5 correct: A=A, B=B, C!=A, C=C, B=B
        self.assertAlmostEqual(results['f1_score'], 0.747, places=2)  # 2 * (0.8 * 0.7) / (0.8 + 0.7)


if __name__ == '__main__':
    unittest.main()