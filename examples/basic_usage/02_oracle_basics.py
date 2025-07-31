#!/usr/bin/env python3
"""
Oracle Basics Example

This example demonstrates the core functionality of the Oracle class,
including metric registration, evaluation, and multi-round workflows.

Run this example:
    python examples/basic_usage/02_oracle_basics.py
"""

import sys
import os
from typing import Dict, Any, List

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from sde_harness.core import Oracle


def example_basic_oracle():
    """Example 1: Basic Oracle setup and single metric"""
    print("🔷 Example 1: Basic Oracle Setup")
    print("-" * 50)

    # Create Oracle instance
    oracle = Oracle()

    # Define a simple accuracy metric
    def accuracy_metric(prediction: str, reference: str, **kwargs) -> float:
        """Simple string matching accuracy"""
        return 1.0 if prediction.lower().strip() == reference.lower().strip() else 0.0

    # Register the metric
    oracle.register_metric("accuracy", accuracy_metric)

    # Test the metric
    test_cases = [
        ("apple", "apple"),  # Perfect match
        ("Apple", "apple"),  # Case difference
        ("apple", "orange"),  # No match
        ("banana ", "banana"),  # Whitespace difference
    ]

    print("Testing accuracy metric:")
    for prediction, reference in test_cases:
        result = oracle.compute(prediction, reference, metrics=["accuracy"])
        print(f"  '{prediction}' vs '{reference}' → {result['accuracy']}")

    print(f"\nRegistered metrics: {oracle.list_metrics()}")
    print()


def example_multiple_metrics():
    """Example 2: Multiple metrics and comprehensive evaluation"""
    print("🔷 Example 2: Multiple Metrics")
    print("-" * 50)

    oracle = Oracle()

    # Define multiple metrics
    def length_similarity(prediction: str, reference: str, **kwargs) -> float:
        """Similarity based on length difference"""
        if len(reference) == 0:
            return 1.0 if len(prediction) == 0 else 0.0

        diff = abs(len(prediction) - len(reference))
        max_len = max(len(prediction), len(reference))
        return max(0.0, 1.0 - (diff / max_len))

    def word_overlap(prediction: str, reference: str, **kwargs) -> float:
        """Jaccard similarity of words"""
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())

        if len(pred_words) == 0 and len(ref_words) == 0:
            return 1.0
        if len(pred_words) == 0 or len(ref_words) == 0:
            return 0.0

        intersection = len(pred_words.intersection(ref_words))
        union = len(pred_words.union(ref_words))
        return intersection / union

    def semantic_score(prediction: str, reference: str, **kwargs) -> float:
        """Mock semantic similarity (in practice, use embeddings)"""
        # This is a simplified mock - in real usage, you'd use embeddings
        common_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to"]

        pred_content = [w for w in prediction.lower().split() if w not in common_words]
        ref_content = [w for w in reference.lower().split() if w not in common_words]

        if not pred_content and not ref_content:
            return 1.0
        if not pred_content or not ref_content:
            return 0.0

        common = len(set(pred_content).intersection(set(ref_content)))
        total = len(set(pred_content).union(set(ref_content)))
        return common / total if total > 0 else 0.0

    # Register all metrics
    oracle.register_metric("length_similarity", length_similarity)
    oracle.register_metric("word_overlap", word_overlap)
    oracle.register_metric("semantic_score", semantic_score)

    # Test with various examples
    test_pairs = [
        ("The cat sat on the mat", "The cat sat on the mat"),
        ("A dog ran in the park", "The dog ran in the park"),
        ("Python is great", "Programming languages are useful"),
        ("Short", "This is a much longer sentence"),
    ]

    print("Multi-metric evaluation:")
    for prediction, reference in test_pairs:
        results = oracle.compute(prediction, reference)
        print(f"\nPrediction: '{prediction}'")
        print(f"Reference:  '{reference}'")
        for metric, score in results.items():
            print(f"  {metric}: {score:.3f}")

    print()


def example_custom_metrics_with_parameters():
    """Example 3: Custom metrics with parameters"""
    print("🔷 Example 3: Custom Metrics with Parameters")
    print("-" * 50)

    oracle = Oracle()

    def threshold_similarity(
        prediction: str, reference: str, threshold: float = 0.5, **kwargs
    ) -> float:
        """Binary similarity based on word overlap threshold"""
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())

        if len(pred_words) == 0 and len(ref_words) == 0:
            return 1.0
        if len(pred_words) == 0 or len(ref_words) == 0:
            return 0.0

        overlap = len(pred_words.intersection(ref_words)) / len(
            pred_words.union(ref_words)
        )
        return 1.0 if overlap >= threshold else 0.0

    def length_penalty(
        prediction: str, reference: str, penalty_factor: float = 0.1, **kwargs
    ) -> float:
        """Score with length penalty"""
        base_score = 1.0 if prediction.lower() in reference.lower() else 0.0
        length_diff = abs(len(prediction) - len(reference))
        penalty = length_diff * penalty_factor
        return max(0.0, base_score - penalty)

    oracle.register_metric("threshold_similarity", threshold_similarity)
    oracle.register_metric("length_penalty", length_penalty)

    # Test with different parameters
    prediction = "machine learning algorithms"
    reference = "machine learning and deep learning algorithms"

    print(f"Prediction: '{prediction}'")
    print(f"Reference:  '{reference}'")
    print()

    # Test threshold similarity with different thresholds
    for threshold in [0.3, 0.5, 0.7]:
        result = oracle.compute(
            prediction, reference, metrics=["threshold_similarity"], threshold=threshold
        )
        print(f"Threshold {threshold}: {result['threshold_similarity']}")

    # Test length penalty with different factors
    print()
    for factor in [0.0, 0.05, 0.1]:
        result = oracle.compute(
            prediction, reference, metrics=["length_penalty"], penalty_factor=factor
        )
        print(f"Penalty factor {factor}: {result['length_penalty']:.3f}")

    print()


def example_multi_round_metrics():
    """Example 4: Multi-round metrics with history"""
    print("🔷 Example 4: Multi-round Metrics")
    print("-" * 50)

    oracle = Oracle()

    # Register a basic single-round metric
    def accuracy(prediction: str, reference: str, **kwargs) -> float:
        return 1.0 if prediction.lower().strip() == reference.lower().strip() else 0.0

    oracle.register_metric("accuracy", accuracy)

    # Define multi-round metrics
    def improvement_trend(
        history: Dict, reference: Any, current_iteration: int, **kwargs
    ) -> float:
        """Calculate if accuracy is improving over iterations"""
        if "scores" not in history or len(history["scores"]) < 2:
            return 0.0

        accuracy_scores = [score.get("accuracy", 0.0) for score in history["scores"]]
        if len(accuracy_scores) < 2:
            return 0.0

        # Simple trend: current vs previous
        current_score = accuracy_scores[-1]
        previous_score = accuracy_scores[-2]
        return max(0.0, current_score - previous_score)

    def consistency_score(
        history: Dict, reference: Any, current_iteration: int, **kwargs
    ) -> float:
        """Measure consistency of performance across iterations"""
        if "scores" not in history or len(history["scores"]) < 2:
            return 1.0

        accuracy_scores = [score.get("accuracy", 0.0) for score in history["scores"]]
        if len(accuracy_scores) < 2:
            return 1.0

        # Calculate standard deviation (lower is more consistent)
        mean_score = sum(accuracy_scores) / len(accuracy_scores)
        variance = sum((score - mean_score) ** 2 for score in accuracy_scores) / len(
            accuracy_scores
        )
        std_dev = variance**0.5

        # Convert to consistency score (1 = perfectly consistent, 0 = very inconsistent)
        return max(0.0, 1.0 - std_dev)

    oracle.register_multi_round_metric("improvement_trend", improvement_trend)
    oracle.register_multi_round_metric("consistency_score", consistency_score)

    # Simulate multi-round evaluation
    reference = "correct answer"

    # Simulate iteration history
    simulation_data = [
        ("wrong answer", {"accuracy": 0.0}),
        ("close answer", {"accuracy": 0.0}),
        ("correct answer", {"accuracy": 1.0}),
        ("correct answer", {"accuracy": 1.0}),
    ]

    history = {"outputs": [], "scores": []}

    print("Multi-round evaluation simulation:")
    for i, (prediction, expected_scores) in enumerate(simulation_data, 1):
        print(f"\nIteration {i}:")
        print(f"  Prediction: '{prediction}'")

        # Add to history
        history["outputs"].append(prediction)
        history["scores"].append(expected_scores)

        # Compute multi-round metrics
        results = oracle.compute_with_history(
            prediction=prediction,
            reference=reference,
            history=history,
            current_iteration=i,
        )

        print(f"  Results: {results}")

    print()


def example_batch_evaluation():
    """Example 5: Batch evaluation"""
    print("🔷 Example 5: Batch Evaluation")
    print("-" * 50)

    oracle = Oracle()

    # Simple accuracy metric
    def accuracy(prediction: str, reference: str, **kwargs) -> float:
        return 1.0 if prediction.lower().strip() == reference.lower().strip() else 0.0

    def contains_keyword(prediction: str, reference: str, **kwargs) -> float:
        """Check if prediction contains reference as keyword"""
        return 1.0 if reference.lower() in prediction.lower() else 0.0

    oracle.register_metric("accuracy", accuracy)
    oracle.register_metric("contains_keyword", contains_keyword)

    # Batch data
    predictions = [
        "Paris is the capital of France",
        "London is the capital of England",
        "Berlin is the capital of Germany",
        "Madrid is the capital of Spain",
    ]

    references = [
        "Paris",
        "London",
        "Berlin",
        "Rome",  # Intentional mismatch
    ]

    # Evaluate batch
    batch_results = oracle.evaluate_batch(predictions, references)

    print("Batch evaluation results:")
    print(f"Predictions: {predictions}")
    print(f"References: {references}")
    print()

    for metric, scores in batch_results.items():
        avg_score = sum(scores) / len(scores)
        print(f"{metric}:")
        print(f"  Individual scores: {scores}")
        print(f"  Average score: {avg_score:.3f}")
        print()

    print()


def example_error_handling():
    """Example 6: Error handling in metrics"""
    print("🔷 Example 6: Error Handling")
    print("-" * 50)

    oracle = Oracle()

    # Metric that might raise errors
    def error_prone_metric(prediction: str, reference: str, **kwargs) -> float:
        """Metric that raises error for specific inputs"""
        if prediction == "ERROR":
            raise ValueError("Intentional error for demonstration")

        return len(prediction) / (len(reference) + 1)  # +1 to avoid division by zero

    oracle.register_metric("error_prone", error_prone_metric)

    test_cases = [
        ("normal input", "reference"),
        ("ERROR", "reference"),  # This will cause an error
        ("", "reference"),  # Edge case: empty prediction
    ]

    print("Testing error handling:")
    for prediction, reference in test_cases:
        try:
            result = oracle.compute(prediction, reference, metrics=["error_prone"])
            print(f"  '{prediction}' → {result['error_prone']:.3f}")
        except Exception as e:
            print(f"  '{prediction}' → ❌ Error: {type(e).__name__}: {e}")

    print()


def main():
    """Run all Oracle examples"""
    print("🚀 SDE-Harness Oracle Examples")
    print("=" * 60)
    print()

    example_basic_oracle()
    example_multiple_metrics()
    example_custom_metrics_with_parameters()
    example_multi_round_metrics()
    example_batch_evaluation()
    example_error_handling()

    print("✅ All Oracle examples completed!")
    print("\n💡 Next Steps:")
    print("- Try examples/basic_usage/03_prompt_basics.py")
    print("- Create your own domain-specific metrics")
    print("- Experiment with multi-round evaluation patterns")


if __name__ == "__main__":
    main()
