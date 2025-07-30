#!/usr/bin/env python3
"""
Workflow Basics Example (Mock Version for Testing)

This example demonstrates the core functionality of the Workflow class
using mock data to avoid requiring API keys.

Run this example:
    python examples/basic_usage/04_workflow_basics_mock.py
"""

import sys
import os
import asyncio
from typing import Dict, Any, List
from unittest.mock import MagicMock

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from sde_harness.core import Workflow, Oracle, Prompt


class MockGeneration:
    """Mock generation class for testing"""

    def __init__(self, *args, **kwargs):
        self.responses = [
            "Machine learning is a method of data analysis that automates analytical model building. It uses algorithms that iteratively learn from data, allowing computers to find hidden insights without being explicitly programmed.",
            "To improve this explanation: Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve from experience automatically. It uses statistical techniques to give computers the ability to learn patterns from data without being explicitly programmed for each specific task.",
            "Here's a comprehensive explanation: Machine learning represents a revolutionary approach to problem-solving where computers learn to make predictions or decisions by finding patterns in data. Unlike traditional programming, where humans write specific instructions, machine learning algorithms automatically improve their performance through experience, making them invaluable for tasks like image recognition, natural language processing, and predictive analytics.",
        ]
        self.call_count = 0

    def generate(self, prompt, **kwargs):
        """Mock synchronous generation"""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    async def generate_async(self, prompt, **kwargs):
        """Mock asynchronous generation"""
        return self.generate(prompt, **kwargs)


def example_basic_workflow():
    """Example 1: Basic workflow setup and execution with mock data"""
    print("🔷 Example 1: Basic Workflow (Mock)")
    print("-" * 50)

    try:
        # Setup components with mock
        generator = MockGeneration()

        oracle = Oracle()

        # Define a simple quality metric
        def response_quality(prediction: str, reference: str, **kwargs) -> float:
            """Simple quality metric based on length and content"""
            if not prediction.strip():
                return 0.0

            # Basic quality factors
            has_content = len(prediction.strip()) > 10
            has_punctuation = any(p in prediction for p in ".!?")
            reasonable_length = 20 <= len(prediction) <= 500

            score = 0.0
            if has_content:
                score += 0.4
            if has_punctuation:
                score += 0.3
            if reasonable_length:
                score += 0.3

            return score

        oracle.register_metric("quality", response_quality)

        # Create a simple prompt
        prompt = Prompt(
            custom_template="Explain the concept of machine learning in simple terms suitable for a general audience."
        )

        # Create workflow
        workflow = Workflow(generator=generator, oracle=oracle, max_iterations=1)

        # Run workflow
        result = workflow.run_sync(
            prompt=prompt,
            reference="educational explanation",  # Reference for evaluation
            gen_args={},  # No API args needed for mock
        )

        print("Workflow Results:")
        history = result.get("history", {})
        outputs = history.get("outputs", [])
        if outputs:
            print(f"Generated text: {outputs[0]}")
        print(f"Final scores: {result.get('final_scores', {})}")
        print(
            f"Iterations completed: {result.get('execution_summary', {}).get('total_iterations', 0)}"
        )

    except Exception as e:
        print(f"❌ Error in basic workflow: {e}")
        import traceback

        traceback.print_exc()

    print()


def example_multi_iteration_workflow():
    """Example 2: Multi-iteration workflow with improvement"""
    print("🔷 Example 2: Multi-Iteration Workflow (Mock)")
    print("-" * 50)

    try:
        # Setup components
        generator = MockGeneration()

        oracle = Oracle()

        # Define metrics for iterative improvement
        def clarity_score(prediction: str, reference: str, **kwargs) -> float:
            """Mock clarity score based on simple heuristics"""
            if not prediction.strip():
                return 0.0

            # Simple clarity indicators
            sentences = prediction.split(".")
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(
                len(sentences), 1
            )

            # Prefer moderate sentence length (10-20 words)
            if 10 <= avg_sentence_length <= 20:
                length_score = 1.0
            else:
                length_score = max(0.0, 1.0 - abs(avg_sentence_length - 15) / 15)

            # Check for explanation words
            explanation_words = [
                "because",
                "therefore",
                "thus",
                "since",
                "so",
                "as a result",
            ]
            has_explanations = any(
                word in prediction.lower() for word in explanation_words
            )

            return (length_score * 0.7) + (0.3 if has_explanations else 0.0)

        def improvement_trend(
            history: Dict, reference: Any, current_iteration: int, **kwargs
        ) -> float:
            """Track improvement in clarity over iterations"""
            if "scores" not in history or len(history["scores"]) < 2:
                return 0.0

            clarity_scores = [score.get("clarity", 0.0) for score in history["scores"]]
            if len(clarity_scores) < 2:
                return 0.0

            # Calculate improvement from first to last
            first_score = clarity_scores[0]
            last_score = clarity_scores[-1]

            return max(0.0, last_score - first_score)

        oracle.register_metric("clarity", clarity_score)
        oracle.register_multi_round_metric("improvement", improvement_trend)

        # Create dynamic prompt function
        def create_prompt(iteration: int, history: Dict) -> Prompt:
            if iteration == 1:
                return Prompt(
                    custom_template="""Explain quantum computing in simple terms. Focus on clarity and use examples that a general audience can understand."""
                )
            else:
                previous_output = history["outputs"][-1]
                previous_scores = history["scores"][-1]
                clarity_score = previous_scores.get("clarity", 0.0)

                if clarity_score < 0.5:
                    template = f"""The previous explanation of quantum computing needs improvement. Here's what was written:

"{previous_output}"

Please rewrite this explanation to be clearer and more accessible to a general audience. Use simpler language and better examples."""
                else:
                    template = f"""The previous explanation of quantum computing was good. Here's what was written:

"{previous_output}"

Please expand on this explanation with more concrete examples and applications that make the concept even clearer."""

                return Prompt(custom_template=template)

        # Create workflow with multiple iterations
        workflow = Workflow(
            generator=generator,
            oracle=oracle,
            max_iterations=3,
            enable_multi_round_metrics=True,
        )

        # Run workflow
        result = workflow.run_sync(
            prompt=create_prompt, reference="clear explanation", gen_args={}
        )

        print("Multi-Iteration Workflow Results:")
        history = result.get("history", {})
        outputs = history.get("outputs", [])
        scores = history.get("scores", [])

        for i, (output, score) in enumerate(zip(outputs, scores), 1):
            print(f"\nIteration {i}:")
            print(f"Output: {output[:100]}...")
            print(f"Scores: {score}")

        print(f"\nFinal scores: {result.get('final_scores', {})}")
        print(
            f"Total iterations: {result.get('execution_summary', {}).get('total_iterations', 0)}"
        )

    except Exception as e:
        print(f"❌ Error in multi-iteration workflow: {e}")
        import traceback

        traceback.print_exc()

    print()


def example_workflow_with_stop_criteria():
    """Example 3: Workflow with custom stop criteria"""
    print("🔷 Example 3: Workflow with Stop Criteria (Mock)")
    print("-" * 50)

    try:
        # Setup components
        generator = MockGeneration()

        oracle = Oracle()

        # Quality metric
        def quality_score(prediction: str, reference: str, **kwargs) -> float:
            """Comprehensive quality assessment"""
            if not prediction.strip():
                return 0.0

            # Multiple quality factors
            word_count = len(prediction.split())
            has_structure = any(
                marker in prediction for marker in ["1.", "2.", "-", "•"]
            )
            has_examples = (
                "example" in prediction.lower() or "for instance" in prediction.lower()
            )
            appropriate_length = 100 <= word_count <= 300

            score = 0.0
            if has_structure:
                score += 0.3
            if has_examples:
                score += 0.3
            if appropriate_length:
                score += 0.4

            return score

        oracle.register_metric("quality", quality_score)

        # Custom stop criteria
        def stop_when_good_enough(stop_context: Dict) -> bool:
            """Stop if quality score is above threshold"""
            history = stop_context.get("history", {})
            current_iteration = stop_context.get("current_iteration", 0)

            if not history.get("scores"):
                return False

            latest_scores = history["scores"][-1]
            quality = latest_scores.get("quality", 0.0)

            # Stop if quality is high or we've reached max iterations
            return quality >= 0.8 or current_iteration >= 5

        # Create prompt function that gives feedback
        def adaptive_prompt(iteration: int, history: Dict) -> Prompt:
            if iteration == 1:
                return Prompt(
                    custom_template="""Write a comprehensive guide on effective time management. Include structured points and practical examples."""
                )
            else:
                previous_output = history["outputs"][-1]
                previous_scores = history["scores"][-1]
                quality = previous_scores.get("quality", 0.0)

                feedback = ""
                if quality < 0.3:
                    feedback = "The response lacks structure and examples."
                elif quality < 0.6:
                    feedback = (
                        "The response needs better organization or more examples."
                    )
                else:
                    feedback = "The response is good but could be improved."

                template = f"""Previous response: "{previous_output[:100]}..."

Feedback: {feedback}

Please improve the guide on time management based on this feedback. Make sure to include:
1. Clear structure with numbered points or bullet points
2. Practical examples that people can relate to
3. Appropriate length (100-300 words)"""

                return Prompt(custom_template=template)

        # Create workflow with custom stop criteria
        workflow = Workflow(
            generator=generator,
            oracle=oracle,
            max_iterations=5,
            stop_criteria=stop_when_good_enough,
        )

        # Run workflow
        result = workflow.run_sync(
            prompt=adaptive_prompt, reference="time management guide", gen_args={}
        )

        print("Workflow with Stop Criteria Results:")
        history = result.get("history", {})
        outputs = history.get("outputs", [])
        scores = history.get("scores", [])

        total_iterations = result.get("execution_summary", {}).get(
            "total_iterations", 0
        )
        print(f"Completed iterations: {total_iterations}")
        print(
            f"Final quality score: {result.get('final_scores', {}).get('quality', 'N/A')}"
        )
        print(f"Stopped early: {'Yes' if total_iterations < 5 else 'No'}")

        # Show progression
        for i, score in enumerate(scores, 1):
            quality = score.get("quality", 0.0)
            print(f"Iteration {i}: Quality = {quality:.3f}")

        if outputs:
            print(f"\nFinal output: {outputs[-1][:200]}...")

    except Exception as e:
        print(f"❌ Error in workflow with stop criteria: {e}")
        import traceback

        traceback.print_exc()

    print()


def main():
    """Run all Workflow examples with mock data"""
    print("🚀 SDE-Harness Workflow Examples (Mock Version)")
    print("=" * 70)
    print("ℹ️  This version uses mock data to demonstrate workflow patterns")
    print("   without requiring API keys or internet connection.")
    print()

    example_basic_workflow()
    example_multi_iteration_workflow()
    example_workflow_with_stop_criteria()

    print("✅ All Workflow examples completed!")
    print("\n💡 Next Steps:")
    print("- Set up real API keys in config/credentials.yaml")
    print("- Try the real workflow example: 04_workflow_basics.py")
    print("- Experiment with different stop criteria")
    print("- Create domain-specific workflows")


if __name__ == "__main__":
    main()
