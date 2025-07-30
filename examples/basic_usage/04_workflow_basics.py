#!/usr/bin/env python3
"""
Workflow Basics Example

This example demonstrates the core functionality of the Workflow class,
showing how to combine Generation, Oracle, and Prompt into complete workflows.

Run this example:
    python examples/basic_usage/04_workflow_basics.py
"""

import sys
import os
import asyncio
from typing import Dict, Any, List

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from sde_harness.core import Workflow, Generation, Oracle, Prompt


def example_basic_workflow():
    """Example 1: Basic workflow setup and execution"""
    print("🔷 Example 1: Basic Workflow")
    print("-" * 50)

    try:
        # Setup components
        generator = Generation(
            models_file=f"{project_root}/models.yaml",
            credentials_file=f"{project_root}/credentials.yaml",
        )

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
            custom_template="Explain the concept of {topic} in simple terms suitable for a general audience."
        )
        
        # Build the prompt with variables
        built_prompt = prompt.build({"topic": "machine learning"})

        # Create workflow
        workflow = Workflow(generator=generator, oracle=oracle, max_iterations=1)

        # Run workflow
        result = workflow.run_sync(
            prompt=prompt,  # Pass the Prompt object, not the built string
            reference="educational explanation",  # Reference for evaluation
            gen_args={
                "model_name": "openai/gpt-4o-mini",
                "max_tokens": 200,
                "temperature": 0.7,
            },
        )

        print("Workflow Results:")
        history = result.get("history", {})
        outputs = history.get("outputs", [])
        if outputs:
            print(f"Generated text: {outputs[0]}")
        print(f"Final scores: {result['final_scores']}")
        print(f"Iterations completed: {len(outputs)}")

    except Exception as e:
        print(f"❌ Error in basic workflow: {e}")
        print("💡 Check your configuration files and API keys")

    print()


def example_multi_iteration_workflow():
    """Example 2: Multi-iteration workflow with improvement"""
    print("🔷 Example 2: Multi-Iteration Workflow")
    print("-" * 50)

    try:
        # Setup components
        generator = Generation(
            models_file=f"{project_root}/models.yaml",
            credentials_file=f"{project_root}/credentials.yaml",
        )

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
                prompt_text = """Explain quantum computing in simple terms. Focus on clarity and use examples that a general audience can understand."""
            else:
                previous_output = history["outputs"][-1]
                previous_scores = history["scores"][-1]
                clarity_score = previous_scores.get("clarity", 0.0)

                if clarity_score < 0.5:
                    prompt_text = f"""The previous explanation of quantum computing needs improvement. Here's what was written:

"{previous_output}"

Please rewrite this explanation to be clearer and more accessible to a general audience. Use simpler language and better examples."""
                else:
                    prompt_text = f"""The previous explanation of quantum computing was good. Here's what was written:

"{previous_output}"

Please expand on this explanation with more concrete examples and applications that make the concept even clearer."""
            
            return Prompt(custom_template=prompt_text)

        # Create workflow with multiple iterations
        workflow = Workflow(
            generator=generator,
            oracle=oracle,
            max_iterations=3,
            enable_multi_round_metrics=True,
        )

        # Run workflow
        result = workflow.run_sync(
            prompt=create_prompt,
            reference="clear explanation",
            gen_args={
                "model_name": "openai/gpt-4o-mini",
                "max_tokens": 300,
                "temperature": 0.6,
            },
        )

        print("Multi-Iteration Workflow Results:")
        history = result.get("history", {})
        outputs = history.get("outputs", [])
        scores_list = history.get("scores", [])
        
        for i, (output, scores) in enumerate(
            zip(outputs, scores_list), 1
        ):
            print(f"\nIteration {i}:")
            print(f"Output: {output[:100]}...")
            print(f"Scores: {scores}")

        print(f"\nFinal scores: {result['final_scores']}")
        print(f"Total iterations: {len(outputs)}")

    except Exception as e:
        print(f"❌ Error in multi-iteration workflow: {e}")

    print()


async def example_async_workflow():
    """Example 3: Asynchronous workflow execution"""
    print("🔷 Example 3: Async Workflow")
    print("-" * 50)

    try:
        # Setup components
        generator = Generation(
            models_file=f"{project_root}/models.yaml",
            credentials_file=f"{project_root}/credentials.yaml",
        )

        oracle = Oracle()

        # Simple relevance metric
        def relevance_score(prediction: str, reference: str, **kwargs) -> float:
            """Check if prediction is relevant to the topic"""
            topic_keywords = reference.lower().split()
            prediction_lower = prediction.lower()

            matches = sum(
                1 for keyword in topic_keywords if keyword in prediction_lower
            )
            return min(1.0, matches / len(topic_keywords))

        oracle.register_metric("relevance", relevance_score)

        # Create workflow
        workflow = Workflow(generator=generator, oracle=oracle, max_iterations=2)

        # Multiple topics to process concurrently
        topics = [
            "artificial intelligence machine learning",
            "climate change environmental science",
            "quantum physics quantum computing",
        ]

        # Create async tasks
        async def process_topic(topic: str) -> Dict[str, Any]:
            prompt_text = f"Write a brief introduction to {topic} for students."
            prompt = Prompt(custom_template=prompt_text)

            result = await workflow.run(
                prompt=prompt,
                reference=topic,  # Use topic as reference for relevance scoring
                gen_args={
                    "model_name": "openai/gpt-4o-mini",
                    "max_tokens": 150,
                    "temperature": 0.5,
                },
            )

            return {"topic": topic, "result": result}

        # Run multiple workflows concurrently
        tasks = [process_topic(topic) for topic in topics]
        results = await asyncio.gather(*tasks)

        print("Async Workflow Results:")
        for topic_result in results:
            topic = topic_result["topic"]
            result = topic_result["result"]

            print(f"\nTopic: {topic}")
            if 'outputs' in result and result['outputs']:
                print(f"Generated: {result['outputs'][0][:100]}...")
            elif 'history' in result and 'outputs' in result['history'] and result['history']['outputs']:
                print(f"Generated: {result['history']['outputs'][0][:100]}...")
            
            if 'final_scores' in result:
                print(f"Relevance Score: {result['final_scores'].get('relevance', 'N/A')}")
            elif 'history' in result and 'scores' in result['history'] and result['history']['scores']:
                print(f"Relevance Score: {result['history']['scores'][-1].get('relevance', 'N/A')}")

    except Exception as e:
        print(f"❌ Error in async workflow: {e}")

    print()


def example_workflow_with_stop_criteria():
    """Example 4: Workflow with custom stop criteria"""
    print("🔷 Example 4: Workflow with Stop Criteria")
    print("-" * 50)

    try:
        # Setup components
        generator = Generation(
            models_file=f"{project_root}/models.yaml",
            credentials_file=f"{project_root}/credentials.yaml",
        )

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
        def stop_when_good_enough(context: Dict[str, Any]) -> bool:
            """Stop if quality score is above threshold"""
            scores = context.get("scores", {})
            iteration = context.get("iteration", 0)
            
            quality = scores.get("quality", 0.0)

            # Stop if quality is high
            return quality >= 0.8

        # Create prompt function that gives feedback
        def adaptive_prompt(iteration: int, history: Dict) -> Prompt:
            if iteration == 1:
                prompt_text = """Write a comprehensive guide on effective time management. Include structured points and practical examples."""
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

                prompt_text = f"""Previous response: "{previous_output[:100]}..."

Feedback: {feedback}

Please improve the guide on time management based on this feedback. Make sure to include:
1. Clear structure with numbered points or bullet points
2. Practical examples that people can relate to
3. Appropriate length (100-300 words)"""
            
            return Prompt(custom_template=prompt_text)

        # Create workflow with custom stop criteria
        workflow = Workflow(
            generator=generator,
            oracle=oracle,
            max_iterations=5,
            stop_criteria=stop_when_good_enough,
        )

        # Run workflow
        result = workflow.run_sync(
            prompt=adaptive_prompt,
            reference="time management guide",
            gen_args={
                "model_name": "openai/gpt-4o-mini",
                "max_tokens": 400,
                "temperature": 0.7,
            },
        )

        print("Workflow with Stop Criteria Results:")
        
        # Handle different result formats
        if 'outputs' in result:
            outputs = result['outputs']
        elif 'history' in result and 'outputs' in result['history']:
            outputs = result['history']['outputs']
        else:
            outputs = []
            
        print(f"Completed iterations: {len(outputs)}")
        
        if 'final_scores' in result:
            print(f"Final quality score: {result['final_scores'].get('quality', 'N/A')}")
        elif 'history' in result and 'scores' in result['history'] and result['history']['scores']:
            print(f"Final quality score: {result['history']['scores'][-1].get('quality', 'N/A')}")
            
        print(f"Stopped early: {'Yes' if len(outputs) < 5 else 'No'}")

        # Show progression
        if 'iteration_scores' in result:
            for i, scores in enumerate(result["iteration_scores"], 1):
                quality = scores.get("quality", 0.0)
                print(f"Iteration {i}: Quality = {quality:.3f}")
        elif 'history' in result and 'scores' in result['history']:
            for i, scores in enumerate(result['history']['scores'], 1):
                quality = scores.get("quality", 0.0)
                print(f"Iteration {i}: Quality = {quality:.3f}")

        if outputs:
            print(f"\nFinal output: {outputs[-1][:200]}...")

    except Exception as e:
        print(f"❌ Error in workflow with stop criteria: {e}")

    print()


def example_complex_workflow():
    """Example 5: Complex workflow with multiple metrics and analysis"""
    print("🔷 Example 5: Complex Workflow")
    print("-" * 50)

    try:
        # Setup components
        generator = Generation(
            models_file=f"{project_root}/models.yaml",
            credentials_file=f"{project_root}/credentials.yaml",
        )

        oracle = Oracle()

        # Multiple evaluation metrics
        def completeness(prediction: str, reference: str, **kwargs) -> float:
            """Check if response covers required topics"""
            required_topics = ["definition", "examples", "applications", "benefits"]
            prediction_lower = prediction.lower()

            covered = sum(1 for topic in required_topics if topic in prediction_lower)
            return covered / len(required_topics)

        def technical_accuracy(prediction: str, reference: str, **kwargs) -> float:
            """Mock technical accuracy check"""
            # In practice, this would involve more sophisticated checking
            technical_terms = ["algorithm", "data", "model", "system", "process"]
            prediction_lower = prediction.lower()

            terms_used = sum(1 for term in technical_terms if term in prediction_lower)
            return min(1.0, terms_used / 3)  # At least 3 technical terms

        def consistency_metric(
            history: Dict, reference: Any, current_iteration: int, **kwargs
        ) -> float:
            """Multi-round: Check consistency across iterations"""
            if "outputs" not in history or len(history["outputs"]) < 2:
                return 1.0

            # Simple consistency check: similar key terms usage
            current_output = history["outputs"][-1].lower()
            previous_output = history["outputs"][-2].lower()

            current_words = set(current_output.split())
            previous_words = set(previous_output.split())

            overlap = len(current_words.intersection(previous_words))
            union = len(current_words.union(previous_words))

            return overlap / union if union > 0 else 0.0

        # Register metrics
        oracle.register_metric("completeness", completeness)
        oracle.register_metric("technical_accuracy", technical_accuracy)
        oracle.register_multi_round_metric("consistency", consistency_metric)

        # Complex prompt with context
        def research_prompt(iteration: int, history: Dict) -> Prompt:
            base_context = """You are writing a technical report on machine learning applications in healthcare."""

            if iteration == 1:
                prompt_text = f"""{base_context}

Write the introduction section covering:
- Definition of machine learning in healthcare context
- Key applications and use cases
- Benefits and advantages
- Include specific examples
"""
            else:
                previous_output = history["outputs"][-1]
                previous_scores = history["scores"][-1]

                # Identify areas needing improvement
                improvements_needed = []
                if previous_scores.get("completeness", 0) < 0.7:
                    improvements_needed.append(
                        "missing key topics (definition, examples, applications, benefits)"
                    )
                if previous_scores.get("technical_accuracy", 0) < 0.6:
                    improvements_needed.append("needs more technical terminology")

                improvement_text = (
                    "; ".join(improvements_needed)
                    if improvements_needed
                    else "minor refinements"
                )

                prompt_text = f"""{base_context}

Previous version: "{previous_output[:150]}..."

The previous version needs improvement: {improvement_text}

Please revise the introduction section to better address these areas while maintaining consistency with the previous version."""
            
            return Prompt(custom_template=prompt_text)

        # Create complex workflow
        workflow = Workflow(
            generator=generator,
            oracle=oracle,
            max_iterations=3,
            enable_multi_round_metrics=True,
        )

        # Run workflow
        result = workflow.run_sync(
            prompt=research_prompt,
            reference="healthcare AI report",
            gen_args={
                "model_name": "openai/gpt-4o-mini",
                "max_tokens": 500,
                "temperature": 0.6,
            },
        )

        print("Complex Workflow Results:")
        print("=" * 40)

        # Detailed analysis
        history = result.get("history", {})
        outputs = history.get("outputs", [])
        scores_list = history.get("scores", [])
        
        for i, (output, scores) in enumerate(
            zip(outputs, scores_list), 1
        ):
            print(f"\nIteration {i}:")
            print(f"Completeness: {scores.get('completeness', 0):.3f}")
            print(f"Technical Accuracy: {scores.get('technical_accuracy', 0):.3f}")
            if "consistency" in scores:
                print(f"Consistency: {scores.get('consistency', 0):.3f}")

            print(f"Output preview: {output[:150]}...")
            print("-" * 30)

        # Final summary
        final_scores = result["final_scores"]
        print(f"\nFinal Summary:")
        print(f"Total iterations: {len(outputs)}")
        print(f"Final completeness: {final_scores.get('completeness', 0):.3f}")
        print(
            f"Final technical accuracy: {final_scores.get('technical_accuracy', 0):.3f}"
        )
        if "consistency" in final_scores:
            print(f"Final consistency: {final_scores.get('consistency', 0):.3f}")

        # Calculate overall quality
        overall_quality = sum(final_scores.values()) / len(final_scores)
        print(f"Overall quality score: {overall_quality:.3f}")

    except Exception as e:
        print(f"❌ Error in complex workflow: {e}")

    print()


def main():
    """Run all Workflow examples"""
    print("🚀 SDE-Harness Workflow Examples")
    print("=" * 60)
    print()

    example_basic_workflow()
    example_multi_iteration_workflow()
    example_workflow_with_stop_criteria()
    example_complex_workflow()

    # Run async example
    print("Running async example...")
    asyncio.run(example_async_workflow())

    print("✅ All Workflow examples completed!")
    print("\n💡 Next Steps:")
    print("- Try the advanced usage examples")
    print("- Experiment with different stop criteria")
    print("- Create domain-specific workflows")
    print("- Combine all components for your use case")


if __name__ == "__main__":
    main()
