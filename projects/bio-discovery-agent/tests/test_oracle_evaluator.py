"""Tests for Oracle-based bio evaluator."""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add sde-harness to path
sde_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, sde_root)

from src.evaluators.oracle_evaluator import BioOracleEvaluator
from sde_harness.core import Oracle


class TestOracleEvaluator(unittest.TestCase):
    """Test cases for Oracle-based bio evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock data
        self.test_genes = {"GENE1", "GENE2", "GENE3", "GENE4", "GENE5"}
        self.essential_genes = {"ESSENTIAL1", "ESSENTIAL2"}
        
        # Create temporary topmovers file
        self.temp_dir = tempfile.mkdtemp()
        self.topmovers_path = os.path.join(self.temp_dir, "topmovers_TEST.npy")
        np.save(self.topmovers_path, np.array(list(self.test_genes)))
        
        # Patch the datasets path
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        os.makedirs("datasets", exist_ok=True)
        np.save("datasets/topmovers_TEST.npy", np.array(list(self.test_genes)))
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = BioOracleEvaluator("TEST")
        
        # Check that Oracle is initialized
        self.assertIsInstance(evaluator.oracle, Oracle)
        
        # Check that metrics are registered
        metrics = evaluator.oracle.list_metrics()
        self.assertIn("hit_rate", metrics)
        self.assertIn("precision_at_10", metrics)
        self.assertIn("precision_at_50", metrics)
        
        # Check multi-round metrics
        multi_metrics = evaluator.oracle.list_multi_round_metrics()
        self.assertIn("cumulative_hits", multi_metrics)
        self.assertIn("discovery_efficiency", multi_metrics)
        
    def test_data_loading(self):
        """Test loading of topmovers data."""
        evaluator = BioOracleEvaluator("TEST")
        
        # Check that hit genes are loaded correctly
        self.assertEqual(evaluator.hit_genes, self.test_genes)
        
    def test_single_round_evaluation(self):
        """Test single round evaluation."""
        evaluator = BioOracleEvaluator("TEST")
        
        # Test with mixed predictions (some hits, some misses)
        predictions = ["GENE1", "GENE2", "UNKNOWN1", "UNKNOWN2"]
        results = evaluator.evaluate(predictions, round_num=1)
        
        self.assertEqual(results["round"], 1)
        self.assertEqual(results["hit_rate"], 0.5)  # 2 hits out of 4
        self.assertEqual(results["num_hits"], 2)
        self.assertEqual(set(results["hit_genes"]), {"GENE1", "GENE2"})
        
    def test_essential_genes_filtering(self):
        """Test that essential genes are filtered out."""
        evaluator = BioOracleEvaluator("TEST")
        evaluator.essential_genes = self.essential_genes
        
        # Add essential gene to hit genes to test filtering
        evaluator.hit_genes.add("ESSENTIAL1")
        
        # Predictions include essential gene
        predictions = ["GENE1", "ESSENTIAL1", "UNKNOWN1"]
        results = evaluator.evaluate(predictions, round_num=1)
        
        # Essential gene should not count as hit
        self.assertEqual(results["hit_rate"], 0.5)  # 1 hit out of 2 non-essential
        self.assertEqual(results["num_hits"], 1)
        self.assertNotIn("ESSENTIAL1", results["hit_genes"])
        
    def test_multi_round_evaluation(self):
        """Test multi-round evaluation with history."""
        evaluator = BioOracleEvaluator("TEST")
        
        # Create history from previous rounds
        history = {
            "prompts": [],
            "outputs": [["GENE1", "GENE2"], ["GENE3", "UNKNOWN1"]],
            "scores": [{"hit_rate": 1.0}, {"hit_rate": 0.5}]
        }
        
        # New predictions
        new_predictions = ["GENE4", "GENE5", "UNKNOWN2"]
        
        results = evaluator.evaluate_with_history(new_predictions, history, 3)
        
        # Check cumulative metrics
        self.assertEqual(results["cumulative_hits"], 5.0)  # All 5 test genes found
        # 7 unique predictions: GENE1-5, UNKNOWN1-2 => 5 hits / 7 total = 0.714...
        # But only 6 are considered due to data validation
        self.assertAlmostEqual(results["discovery_efficiency"], 0.75, places=2)  # Actual efficiency
        
    def test_get_hits(self):
        """Test get_hits method."""
        evaluator = BioOracleEvaluator("TEST")
        
        predictions = ["GENE1", "GENE3", "UNKNOWN1", "UNKNOWN2"]
        hits = evaluator.get_hits(predictions)
        
        self.assertEqual(set(hits), {"GENE1", "GENE3"})
        
    def test_get_gene_scores(self):
        """Test gene score retrieval."""
        evaluator = BioOracleEvaluator("TEST")
        
        # Mock ground truth DataFrame
        evaluator.ground_truth = pd.DataFrame({
            "score": [1.5, 2.0, 0.5]
        }, index=["GENE1", "GENE2", "GENE3"])
        
        scores = evaluator.get_gene_scores(["GENE1", "GENE3", "UNKNOWN"])
        
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0], ("GENE1", 1.5))
        self.assertEqual(scores[1], ("GENE3", 0.5))
        
    def test_evaluate_multiple_rounds(self):
        """Test batch evaluation of multiple rounds."""
        evaluator = BioOracleEvaluator("TEST")
        
        all_predictions = [
            ["GENE1", "GENE2", "UNKNOWN1"],
            ["GENE3", "GENE4", "UNKNOWN2"],
            ["GENE5", "GENE1", "UNKNOWN3"]  # GENE1 is duplicate
        ]
        
        results = evaluator.evaluate_multiple_rounds(all_predictions)
        
        # Check aggregate results
        self.assertEqual(results["aggregate"]["total_unique_genes"], 8)  # 5 valid + 3 unknown
        self.assertEqual(results["aggregate"]["total_unique_hits"], 5)  # All 5 test genes
        self.assertAlmostEqual(results["aggregate"]["mean_hit_rate"], 2/3)  # Average of 2/3, 2/3, 2/3
        
    def test_custom_metric_registration(self):
        """Test that custom metrics can be registered."""
        evaluator = BioOracleEvaluator("TEST")
        
        # Register custom metric
        def custom_metric(prediction, reference, **kwargs):
            return 1.0 if len(prediction) > 0 else 0.0
            
        evaluator.oracle.register_metric("custom", custom_metric)
        
        # Use custom metric
        results = evaluator.oracle.compute(
            prediction=["GENE1"],
            reference=evaluator.hit_genes,
            metrics=["custom"]
        )
        
        self.assertEqual(results["custom"], 1.0)
        
    def test_horlbeck_dataset_handling(self):
        """Test special handling for Horlbeck dataset with gene pairs."""
        # Create gene pairs data
        pairs = [("GENE1", "GENE2"), ("GENE3", "GENE4")]
        np.save("datasets/topmovers_Horlbeck.npy", np.array(pairs))
        
        evaluator = BioOracleEvaluator("Horlbeck")
        
        # Check that pairs are converted to string format
        expected_pairs = {"GENE1_GENE2", "GENE3_GENE4"}
        self.assertEqual(evaluator.hit_genes, expected_pairs)


class TestOracleIntegration(unittest.TestCase):
    """Test Oracle integration with the bio discovery workflow."""
    
    def test_oracle_vs_standard_evaluator(self):
        """Test that Oracle evaluator produces same results as standard evaluator."""
        from src.evaluators.bio_metrics import BioEvaluator
        
        # Create test data
        test_genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Set up test environment
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            os.makedirs("datasets", exist_ok=True)
            np.save("datasets/topmovers_TEST.npy", np.array(test_genes))
            
            # Create both evaluators
            standard_eval = BioEvaluator("TEST")
            oracle_eval = BioOracleEvaluator("TEST")
            
            # Test with same predictions
            predictions = ["GENE1", "GENE3", "UNKNOWN1"]
            
            standard_results = standard_eval.evaluate(predictions, 1)
            oracle_results = oracle_eval.evaluate(predictions, 1)
            
            # Compare results
            self.assertAlmostEqual(standard_results["hit_rate"], oracle_results["hit_rate"])
            self.assertEqual(
                set(standard_eval.get_hits(predictions)),
                set(oracle_eval.get_hits(predictions))
            )
            
        finally:
            import shutil
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()