"""
Examples demonstrating the enhanced Workflow and Oracle classes
with history support and multi-round metrics.
"""

import asyncio
import os
import sys
from sci_demo.generation import Generation
from sci_demo.workflow import Workflow
from sci_demo.oracle import Oracle, improvement_rate_metric, consistency_metric, convergence_metric
from sci_demo.prompt import Prompt


def setup_components():
    """Setup the basic components for the examples."""
    # Initialize generation with multiple providers
    gen = Generation(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        claude_api_key=os.getenv("CLAUDE_API_KEY"),
    )
    
    # Setup oracle with both single-round and multi-round metrics
    oracle = Oracle()
    
    # Register single-round metrics
    def accuracy_metric(pred, ref, **kwargs):
        """Simple accuracy based on exact match (case-insensitive)."""
        return float(pred.strip().lower() == ref.strip().lower())
    
    def relevance_metric(pred, ref, **kwargs):
        """Simple relevance based on keyword overlap."""
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if not ref_words:
            return 0.0
        return len(pred_words.intersection(ref_words)) / len(ref_words)
    
    def length_quality_metric(pred, ref, **kwargs):
        """Quality based on appropriate length (not too short, not too long)."""
        pred_len = len(pred.split())
        ref_len = len(ref.split())
        if ref_len == 0:
            return 1.0 if pred_len == 0 else 0.0
        ratio = pred_len / ref_len
        # Optimal ratio is between 0.8 and 1.5
        if 0.8 <= ratio <= 1.5:
            return 1.0
        elif ratio < 0.8:
            return ratio / 0.8
        else:
            return 1.5 / ratio
    
    oracle.register_metric("accuracy", accuracy_metric)
    oracle.register_metric("relevance", relevance_metric)
    oracle.register_metric("length_quality", length_quality_metric)
    
    # Register multi-round metrics
    oracle.register_multi_round_metric("improvement_rate", improvement_rate_metric)
    oracle.register_multi_round_metric("consistency", consistency_metric)
    oracle.register_multi_round_metric("convergence", convergence_metric)
    
    return gen, oracle


def example_1_basic_history_prompt():
    """Example 1: Basic usage of history-aware prompts."""
    print("=== Example 1: Basic History-Aware Prompts ===")
    
    gen, oracle = setup_components()
    
    # Create a workflow with history support
    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=4,
        enable_history_in_prompts=True,
        enable_multi_round_metrics=True
    )
    
    # Create an iterative prompt
    prompt = workflow.create_iterative_prompt(
        task_description="Write a concise scientific summary about quantum computing",
        input_text="Quantum computing uses quantum mechanical phenomena to process information",
        template_type="iterative_with_feedback"
    )
    
    # Define stopping criteria
    def stop_when_good_enough(context):
        scores = context["scores"]
        return scores.get("relevance", 0) >= 0.8 and scores.get("length_quality", 0) >= 0.9
    
    workflow.stop_criteria = stop_when_good_enough
    
    # Run the workflow
    result = workflow.run_sync(
        prompt=prompt,
        reference="Quantum computing leverages quantum mechanics for computational advantages in specific problems",
        gen_args={"model": "gpt-4o", "max_tokens": 100, "temperature": 0.7},
        history_context={"task_description": "Scientific summarization with iterative improvement"}
    )
    
    print(f"Completed {result['total_iterations']} iterations")
    print(f"Final scores: {result['final_scores']}")
    print(f"Best iteration: {result['best_iteration']['iteration']} with score {result['best_iteration']['score']:.3f}")
    
    # Show the evolution of outputs
    print("\n--- Output Evolution ---")
    for i, output in enumerate(result['history']['outputs'], 1):
        scores = result['history']['scores'][i-1]
        print(f"Iteration {i}: {output[:80]}...")
        print(f"  Scores: {scores}")
        print()


def example_2_dynamic_prompts():
    """Example 2: Dynamic prompts that change based on iteration and history."""
    print("=== Example 2: Dynamic Prompts with Iteration-Specific Instructions ===")
    
    gen, oracle = setup_components()
    
    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=5,
        enable_history_in_prompts=True,
        enable_multi_round_metrics=True
    )
    
    # Create a dynamic prompt function
    dynamic_prompt = workflow.create_dynamic_prompt_function(
        base_task="Explain the concept of machine learning",
        base_input="Machine learning is a subset of artificial intelligence",
        iteration_instructions={
            1: "Provide a basic explanation suitable for beginners",
            2: "Add more technical details and examples",
            3: "Include real-world applications and use cases",
            4: "Ensure the explanation is both comprehensive and accessible",
            5: "Polish the explanation for clarity and flow"
        }
    )
    
    # Custom stopping criteria based on multi-round metrics
    def stop_when_converged(context):
        scores = context["scores"]
        iteration = context["iteration"]
        
        # Stop if we have good convergence and consistency
        if iteration >= 3:
            convergence = scores.get("convergence", 0)
            consistency = scores.get("consistency", 0)
            if convergence >= 0.7 and consistency >= 0.8:
                return True
        
        return False
    
    workflow.stop_criteria = stop_when_converged
    
    result = workflow.run_sync(
        prompt=dynamic_prompt,
        reference="Machine learning enables computers to learn and improve from data without explicit programming",
        gen_args={"model": "gpt-4o", "max_tokens": 150, "temperature": 0.6},
        history_context={"task_description": "Educational explanation with progressive enhancement"}
    )
    
    print(f"Completed {result['total_iterations']} iterations")
    print(f"Convergence achieved: {result['final_scores'].get('convergence', 0):.3f}")
    print(f"Consistency achieved: {result['final_scores'].get('consistency', 0):.3f}")
    
    # Analyze trends
    print("\n--- Trend Analysis ---")
    trends = result['trend_analysis']
    for metric, trend_data in trends.items():
        if isinstance(trend_data, dict) and 'improvement_rate' in trend_data:
            print(f"{metric}:")
            print(f"  Improvement rate: {trend_data['improvement_rate']:.4f}")
            print(f"  Total improvement: {trend_data['total_improvement']:.4f}")
            print(f"  Best score: {trend_data['best_score']:.4f}")


def example_3_conversation_style():
    """Example 3: Conversation-style iterative improvement."""
    print("=== Example 3: Conversation-Style Iterative Improvement ===")
    
    gen, oracle = setup_components()
    
    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=4,
        enable_history_in_prompts=True,
        enable_multi_round_metrics=False  # Focus on single-round metrics for this example
    )
    
    # Create a conversation-style prompt
    conversation_prompt = Prompt(
        template_name="conversation",
        default_vars={
            "current_message": "Explain photosynthesis in simple terms"
        }
    )
    
    # Custom prompt function that simulates a conversation
    def conversation_prompt_fn(iteration, history):
        if iteration == 1:
            return Prompt(
                template_name="conversation",
                default_vars={
                    "current_message": "Explain photosynthesis in simple terms",
                    "conversation_history": ""
                }
            )
        else:
            # Simulate follow-up questions based on previous responses
            follow_ups = [
                "Can you make it even simpler for a 10-year-old?",
                "What role does chlorophyll play in this process?",
                "How does this relate to climate change?"
            ]
            follow_up = follow_ups[min(iteration-2, len(follow_ups)-1)]
            
            return Prompt(
                template_name="conversation",
                default_vars={
                    "current_message": follow_up
                }
            )
    
    result = workflow.run_sync(
        prompt=conversation_prompt_fn,
        reference="Photosynthesis is how plants make food from sunlight, water, and carbon dioxide",
        gen_args={"model": "gpt-4o", "max_tokens": 120, "temperature": 0.8},
        history_context={"task_description": "Educational conversation about photosynthesis"}
    )
    
    print("--- Conversation Flow ---")
    for i, (prompt, output) in enumerate(zip(result['history']['prompts'], result['history']['outputs']), 1):
        print(f"Turn {i}:")
        # Extract the current message from the prompt
        if "Current message:" in prompt:
            message = prompt.split("Current message:")[-1].split("Please respond")[0].strip()
            print(f"  User: {message}")
        print(f"  Assistant: {output}")
        print(f"  Scores: {result['history']['scores'][i-1]}")
        print()


def example_4_multi_round_metrics_analysis():
    """Example 4: Focus on multi-round metrics and trend analysis."""
    print("=== Example 4: Multi-Round Metrics and Trend Analysis ===")
    
    gen, oracle = setup_components()
    
    # Add a custom multi-round metric
    def diversity_metric(history, reference, current_iteration, **kwargs):
        """Measure diversity of outputs across iterations."""
        if not history.get("outputs") or len(history["outputs"]) < 2:
            return 0.0
        
        outputs = history["outputs"]
        total_pairs = 0
        total_diversity = 0.0
        
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                words1 = set(outputs[i].lower().split())
                words2 = set(outputs[j].lower().split())
                if len(words1) == 0 and len(words2) == 0:
                    diversity = 0.0
                elif len(words1) == 0 or len(words2) == 0:
                    diversity = 1.0
                else:
                    # Jaccard distance (1 - Jaccard similarity)
                    diversity = 1.0 - len(words1.intersection(words2)) / len(words1.union(words2))
                
                total_diversity += diversity
                total_pairs += 1
        
        return total_diversity / total_pairs if total_pairs > 0 else 0.0
    
    oracle.register_multi_round_metric("diversity", diversity_metric)
    
    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=6,
        enable_history_in_prompts=True,
        enable_multi_round_metrics=True
    )
    
    # Create a prompt that encourages variation
    def variation_prompt_fn(iteration, history):
        base_prompt = Prompt(
            template_name="iterative_with_feedback",
            default_vars={
                "task_description": "Write a creative description of a futuristic city",
                "input_text": "A city in the year 2150 with advanced technology",
                "additional_instructions": f"Iteration {iteration}: Try a different perspective or focus area"
            }
        )
        return base_prompt
    
    result = workflow.run_sync(
        prompt=variation_prompt_fn,
        reference="A futuristic city with sustainable technology and harmonious human-AI coexistence",
        gen_args={"model": "gpt-4o", "max_tokens": 100, "temperature": 0.9},  # Higher temperature for creativity
        history_context={"task_description": "Creative writing with variation"}
    )
    
    print("--- Multi-Round Metrics Analysis ---")
    final_scores = result['final_scores']
    print(f"Final improvement rate: {final_scores.get('improvement_rate', 0):.4f}")
    print(f"Final consistency: {final_scores.get('consistency', 0):.4f}")
    print(f"Final convergence: {final_scores.get('convergence', 0):.4f}")
    print(f"Final diversity: {final_scores.get('diversity', 0):.4f}")
    
    # Detailed trend analysis
    print("\n--- Detailed Trend Analysis ---")
    trends = result['trend_analysis']
    for metric_name in ['relevance', 'length_quality']:
        if f"{metric_name}_trends" in trends:
            trend_data = trends[f"{metric_name}_trends"]
            print(f"\n{metric_name.upper()} Trends:")
            for key, value in trend_data.items():
                print(f"  {key}: {value:.4f}")


async def example_5_async_workflow():
    """Example 5: Asynchronous workflow execution."""
    print("=== Example 5: Asynchronous Workflow Execution ===")
    
    gen, oracle = setup_components()
    
    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=3,
        enable_history_in_prompts=True,
        enable_multi_round_metrics=True
    )
    
    prompt = workflow.create_iterative_prompt(
        task_description="Summarize the benefits of renewable energy",
        input_text="Solar, wind, and hydroelectric power are clean energy sources",
        template_type="iterative_with_feedback"
    )
    
    # Run asynchronously
    result = await workflow.run(
        prompt=prompt,
        reference="Renewable energy provides clean, sustainable power while reducing environmental impact",
        gen_args={"model": "gpt-4o", "max_tokens": 80, "temperature": 0.7},
        history_context={"task_description": "Renewable energy summarization"}
    )
    
    print(f"Async execution completed with {result['total_iterations']} iterations")
    print(f"Best output: {result['best_iteration']['output'][:100]}...")


def main():
    """Run all examples."""
    print("=== History-Enhanced Workflow Examples ===\n")
    
    # Check API key availability
    if not any([os.getenv("OPENAI_API_KEY"), os.getenv("GEMINI_API_KEY"), os.getenv("CLAUDE_API_KEY")]):
        print("No API keys found. Please set at least one of:")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("  export GEMINI_API_KEY='your-gemini-key'")
        print("  export CLAUDE_API_KEY='your-claude-key'")
        sys.exit(1)
    
    try:
        # Run synchronous examples
        example_1_basic_history_prompt()
        print("\n" + "="*60 + "\n")
        
        example_2_dynamic_prompts()
        print("\n" + "="*60 + "\n")
        
        example_3_conversation_style()
        print("\n" + "="*60 + "\n")
        
        example_4_multi_round_metrics_analysis()
        print("\n" + "="*60 + "\n")
        
        # Run async example
        print("Running async example...")
        asyncio.run(example_5_async_workflow())
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 