"""Integration tests for Oracle functionality in BioDiscoveryAgent."""
import unittest
import sys
import os
import tempfile
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add sde-harness to path
sde_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, sde_root)

from sde_harness.core import Oracle


class TestOracleDemo(unittest.TestCase):
    """Test Oracle demo functionality."""
    
    def test_bio_oracle_setup(self):
        """Test setting up Oracle with bio-specific metrics."""
        oracle = Oracle()
        
        # Define hit genes for testing
        hit_genes = {"STAT1", "JAK1", "IFNGR1", "IRF1", "SOCS1", "PIAS1"}
        
        # Register hit rate metric
        def hit_rate_metric(prediction, reference, **kwargs):
            if not prediction:
                return 0.0
            hits = sum(1 for gene in prediction if gene in reference)
            return hits / len(prediction)
        
        oracle.register_metric("hit_rate", hit_rate_metric)
        
        # Test single round evaluation
        predicted_genes = ["STAT1", "JAK1", "IFNGR1", "IRF1", "RANDOM_GENE"]
        results = oracle.compute(
            prediction=predicted_genes,
            reference=hit_genes,
            metrics=["hit_rate"]
        )
        
        self.assertAlmostEqual(results["hit_rate"], 0.8)  # 4 hits out of 5
        
    def test_multi_round_evaluation(self):
        """Test multi-round evaluation with history tracking."""
        oracle = Oracle()
        hit_genes = {"STAT1", "JAK1", "IFNGR1", "IRF1", "SOCS1", "PIAS1"}
        
        # Register metrics
        def hit_rate_metric(prediction, reference, **kwargs):
            if not prediction:
                return 0.0
            hits = sum(1 for gene in prediction if gene in reference)
            return hits / len(prediction)
            
        def cumulative_hits_metric(history, reference, current_iteration, prediction=None, **kwargs):
            all_predictions = []
            for output in history.get("outputs", []):
                if isinstance(output, list):
                    all_predictions.extend(output)
            if prediction:
                all_predictions.extend(prediction)
            unique_hits = set(gene for gene in all_predictions if gene in reference)
            return float(len(unique_hits))
        
        oracle.register_metric("hit_rate", hit_rate_metric)
        oracle.register_multi_round_metric("cumulative_hits", cumulative_hits_metric)
        
        # Simulate rounds
        history = {"outputs": [], "scores": []}
        
        # Round 1
        round1_genes = ["STAT1", "JAK1", "IFNGR1", "UNKNOWN1"]
        round1_results = oracle.compute_with_history(
            prediction=round1_genes,
            reference=hit_genes,
            history=history,
            current_iteration=1
        )
        
        self.assertAlmostEqual(round1_results["hit_rate"], 0.75)  # 3/4
        self.assertEqual(round1_results["cumulative_hits"], 3.0)
        
        # Update history
        history["outputs"].append(round1_genes)
        history["scores"].append({"hit_rate": round1_results["hit_rate"]})
        
        # Round 2
        round2_genes = ["IRF1", "SOCS1", "PIAS1", "UNKNOWN2"]
        round2_results = oracle.compute_with_history(
            prediction=round2_genes,
            reference=hit_genes,
            history=history,
            current_iteration=2
        )
        
        self.assertAlmostEqual(round2_results["hit_rate"], 0.75)  # 3/4
        self.assertEqual(round2_results["cumulative_hits"], 6.0)  # All hits found
        
    def test_custom_metrics(self):
        """Test custom metric registration and usage."""
        oracle = Oracle()
        
        # Define novelty metric
        def novelty_metric(prediction, reference, **kwargs):
            seen_genes = kwargs.get('seen_genes', set())
            if not prediction:
                return 0.0
            novel_count = sum(1 for g in prediction if g not in seen_genes)
            return novel_count / len(prediction)
        
        oracle.register_metric("novelty", novelty_metric)
        
        # Test novelty scoring
        predictions = ["STAT1", "JAK1", "NEW_GENE1", "NEW_GENE2"]
        seen = {"STAT1", "JAK1"}
        
        results = oracle.compute(
            prediction=predictions,
            reference=set(),  # Reference not used for novelty
            metrics=["novelty"],
            seen_genes=seen
        )
        
        self.assertAlmostEqual(results["novelty"], 0.5)  # 2 novel out of 4
        
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        oracle = Oracle()
        
        # Create mock history
        history = {
            "scores": [
                {"hit_rate": 0.3},
                {"hit_rate": 0.5},
                {"hit_rate": 0.7}
            ]
        }
        
        # Compute trend metrics
        trends = oracle.compute_trend_metrics(history, "hit_rate")
        
        self.assertIn("improvement_rate", trends)
        self.assertIn("total_improvement", trends)
        self.assertIn("best_score", trends)
        self.assertIn("average_score", trends)
        
        # Check values
        self.assertAlmostEqual(trends["total_improvement"], 0.4)  # 0.7 - 0.3
        self.assertAlmostEqual(trends["best_score"], 0.7)
        self.assertAlmostEqual(trends["average_score"], 0.5)


class TestOracleWithRealData(unittest.TestCase):
    """Test Oracle with simulated real dataset structure."""
    
    def setUp(self):
        """Set up test environment with mock data files."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create datasets directory
        os.makedirs("datasets", exist_ok=True)
        
        # Create mock topmovers data
        test_genes = ["JAK1", "STAT1", "IRF1", "IFNGR1", "SOCS1"]
        np.save("datasets/topmovers_TEST.npy", np.array(test_genes))
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
        
    def test_oracle_evaluator_with_data(self):
        """Test Oracle evaluator with mock data files."""
        from src.evaluators.oracle_evaluator import BioOracleEvaluator
        
        evaluator = BioOracleEvaluator("TEST")
        
        # Verify data loaded correctly
        self.assertEqual(len(evaluator.hit_genes), 5)
        
        # Test evaluation
        predictions = ["JAK1", "STAT1", "UNKNOWN1", "UNKNOWN2"]
        results = evaluator.evaluate(predictions, round_num=1)
        
        self.assertAlmostEqual(results["hit_rate"], 0.5)  # 2/4
        self.assertEqual(results["num_hits"], 2)
        
    def test_multi_round_workflow(self):
        """Test complete multi-round workflow."""
        from src.evaluators.oracle_evaluator import BioOracleEvaluator
        
        evaluator = BioOracleEvaluator("TEST")
        
        # Simulate 3 rounds
        all_predictions = [
            ["JAK1", "STAT1", "UNKNOWN1"],
            ["IRF1", "IFNGR1", "UNKNOWN2"],
            ["SOCS1", "JAK1", "UNKNOWN3"]  # JAK1 is duplicate
        ]
        
        results = evaluator.evaluate_multiple_rounds(all_predictions)
        
        # Check aggregate results
        self.assertEqual(results["aggregate"]["total_unique_hits"], 5)
        self.assertGreater(results["aggregate"]["mean_hit_rate"], 0.5)
        
        # Check final metrics
        self.assertIn("final_metrics", results)
        self.assertEqual(results["final_metrics"]["cumulative_hits"], 5.0)
        self.assertGreater(results["final_metrics"]["discovery_efficiency"], 0)


if __name__ == "__main__":
    unittest.main()