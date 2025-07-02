"""
Comprehensive example demonstrating Weave tracking across the entire science workflow system.

This example shows how to monitor and track:
1. Complete workflow orchestration
2. Individual iterations
3. Model generation performance
4. Oracle evaluations (both single-round and multi-round)
5. Prompt construction and history integration
6. Trend analysis across iterations
7. Performance metrics and improvements

Run this to see comprehensive tracking in your Weave dashboard.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Recording
import weave

from sci_demo.generation import Generation
from sci_demo.workflow import Workflow
from sci_demo.oracle import Oracle, improvement_rate_metric, consistency_metric, convergence_metric
from sci_demo.prompt import Prompt


def setup_comprehensive_tracking_example():
    """Setup a complete example with all tracking features enabled."""
    
    # Initialize weave for the main application
    weave.init("comprehensive_science_workflow")
    
    print("🔬 Setting up Comprehensive Science Workflow with Weave Tracking")
    print("="*60)
    
    # 1. Initialize Generation with tracking (already has @weave.op())
    print("📡 Initializing Generation component...")
    gen = Generation(
        models_file="models.yaml",
        credentials_file="credentials.yaml",
        max_workers=4
    )
    
    # 2. Setup Oracle with comprehensive metrics and tracking
    print("🎯 Setting up Oracle with comprehensive metrics...")
    oracle = Oracle()
    
    # Register single-round metrics
    def accuracy_metric(pred, ref, **kwargs):
        """Scientific accuracy based on key concepts presence."""
        pred_lower = pred.lower()
        ref_lower = ref.lower()
        
        # Extract key scientific terms
        ref_terms = set(ref_lower.split())
        pred_terms = set(pred_lower.split())
        
        if not ref_terms:
            return 1.0 if not pred_terms else 0.0
        
        overlap = len(ref_terms.intersection(pred_terms))
        return overlap / len(ref_terms)
    
    def clarity_metric(pred, ref, **kwargs):
        """Clarity based on sentence structure and length."""
        sentences = pred.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Optimal sentence length: 10-20 words
        if 10 <= avg_length <= 20:
            return 1.0
        elif avg_length < 10:
            return avg_length / 10
        else:
            return 20 / avg_length
    
    def completeness_metric(pred, ref, **kwargs):
        """Completeness based on coverage of reference concepts."""
        ref_words = ref.lower().split()
        pred_words = pred.lower().split()
        
        if not ref_words:
            return 1.0
        
        covered_concepts = sum(1 for word in ref_words if word in pred_words)
        return covered_concepts / len(ref_words)
    
    # Register all single-round metrics
    oracle.register_metric("accuracy", accuracy_metric)
    oracle.register_metric("clarity", clarity_metric)
    oracle.register_metric("completeness", completeness_metric)
    
    # Register multi-round metrics for tracking improvements
    oracle.register_multi_round_metric("improvement_rate", improvement_rate_metric)
    oracle.register_multi_round_metric("consistency", consistency_metric)
    oracle.register_multi_round_metric("convergence", convergence_metric)
    
    # Custom multi-round metric for scientific discourse evolution
    def scientific_sophistication_metric(history, reference, current_iteration, **kwargs):
        """Track how scientific sophistication evolves across iterations."""
        if not history.get("outputs") or len(history["outputs"]) < 2:
            return 0.0
        
        scientific_terms = [
            "hypothesis", "theory", "experiment", "data", "analysis", 
            "evidence", "research", "methodology", "conclusion", "investigation",
            "quantum", "molecular", "genetic", "computational", "statistical"
        ]
        
        current_output = kwargs.get("prediction", "")
        current_score = sum(1 for term in scientific_terms if term in current_output.lower())
        
        # Compare with first iteration
        first_output = history["outputs"][0]
        first_score = sum(1 for term in scientific_terms if term in first_output.lower())
        
        return max(0.0, (current_score - first_score) / len(scientific_terms))
    
    oracle.register_multi_round_metric("scientific_sophistication", scientific_sophistication_metric)
    
    print(f"   ✅ Registered {len(oracle.list_single_round_metrics())} single-round metrics")
    print(f"   ✅ Registered {len(oracle.list_multi_round_metrics())} multi-round metrics")
    
    # 3. Create Workflow with full tracking enabled
    print("🔄 Creating Workflow with comprehensive tracking...")
    
    def intelligent_stop_criteria(context):
        """Smart stopping criteria based on multiple factors."""
        scores = context["scores"]
        iteration = context["iteration"]
        
        # Stop if we achieve high performance across all metrics
        if isinstance(scores, dict):
            avg_score = sum(scores.values()) / len(scores) if scores else 0.0
            if avg_score >= 0.85:
                print(f"   🎯 High performance achieved (avg: {avg_score:.3f})")
                return True
        
        # Stop if improvement rate is very low
        if iteration >= 3 and scores.get("improvement_rate", 0) < 0.01:
            print(f"   📉 Low improvement rate detected")
            return True
        
        # Stop if consistency is very high (converged)
        if iteration >= 2 and scores.get("consistency", 0) > 0.95:
            print(f"   🎲 High consistency detected (converged)")
            return True
        
        return False
    
    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=5,
        stop_criteria=intelligent_stop_criteria,
        enable_history_in_prompts=True,
        enable_multi_round_metrics=True
    )
    
    print("   ✅ Workflow configured with intelligent stopping criteria")
    
    return gen, oracle, workflow


@weave.op()
async def run_comprehensive_tracking_example():
    """Main example function demonstrating comprehensive weave tracking."""
    
    # Setup components
    gen, oracle, workflow = setup_comprehensive_tracking_example()
    
    print("\n🚀 Starting Comprehensive Workflow Execution")
    print("="*60)
    
    # Create dynamic prompt that evolves with iterations
    def create_dynamic_scientific_prompt(iteration: int, history: Dict[str, Any]) -> Prompt:
        """Dynamic prompt that becomes more sophisticated with each iteration."""
        
        base_task = "Write a comprehensive scientific explanation"
        input_topic = "quantum computing applications in drug discovery"
        
        # Iteration-specific instructions
        iteration_instructions = {
            1: "Provide a clear, basic overview focusing on fundamental concepts.",
            2: "Add technical details and explain the underlying mechanisms.",
            3: "Include specific examples and real-world applications with citations.",
            4: "Discuss current limitations and future research directions.",
            5: "Synthesize everything into a coherent, publication-ready summary."
        }
        
        # Choose template based on iteration
        if iteration == 1:
            template_type = "iterative"
        else:
            template_type = "iterative_with_feedback"
        
        # Create prompt with iteration-specific customization
        prompt = Prompt(
            template_name=template_type,
            default_vars={
                "task_description": base_task,
                "input_text": input_topic,
                "additional_instructions": iteration_instructions.get(iteration, "")
            }
        )
        
        # Add sophistication based on iteration
        if iteration > 2:
            prompt.add_vars(
                sophistication_level="advanced",
                expected_length="detailed",
                target_audience="scientific researchers"
            )
        
        return prompt
    
    # Define reference for evaluation
    reference_text = """
    Quantum computing leverages quantum mechanical phenomena like superposition and entanglement 
    to perform computational tasks that are intractable for classical computers. In drug discovery, 
    quantum algorithms can simulate molecular interactions at unprecedented accuracy, enabling 
    researchers to identify promising drug candidates faster and more efficiently. Key applications 
    include protein folding prediction, molecular docking simulations, and optimization of drug 
    properties. Current limitations include quantum decoherence and limited qubit counts, but 
    ongoing research in quantum error correction and hybrid classical-quantum algorithms shows 
    promise for transformative impacts in pharmaceutical research.
    """
    
    # Generation arguments for consistent tracking
    gen_args = {
        "model_name": "gpt-4o",  # Adjust based on your models.yaml
        "max_tokens": 300,
        "temperature": 0.7,
    }
    
    # History context for richer tracking
    history_context = {
        "task_description": "Scientific explanation of quantum computing in drug discovery",
        "experiment_id": "comprehensive_tracking_demo",
        "user": "researcher",
        "domain": "computational_chemistry"
    }
    
    print("🎬 Executing workflow with comprehensive tracking...")
    print(f"   📝 Topic: {gen_args}")
    print(f"   🎯 Reference length: {len(reference_text)} characters")
    print(f"   ⚙️  Generation args: {gen_args}")
    
    # Execute the workflow with full tracking
    try:
        result = await workflow.run(
            prompt=create_dynamic_scientific_prompt,
            reference=reference_text,
            gen_args=gen_args,
            history_context=history_context
        )
        
        print("\n✅ Workflow Execution Complete!")
        print("="*60)
        
        # Display comprehensive results
        print_comprehensive_results(result)
        
        # Log final comprehensive summary
        call = weave.get_current_call()
        if call and call.summary:
            call.summary.update({
                "experiment_complete": {
                    "total_iterations": result["total_iterations"],
                    "final_performance": result["final_scores"],
                    "best_iteration": result["best_iteration"],
                    "experiment_success": True,
                    "key_insights": analyze_experiment_insights(result)
                }
            })
        
        return result
        
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
        call = weave.get_current_call()
        if call and call.summary:
            call.summary.update({
                "experiment_error": {
                    "error_message": str(e),
                    "error_type": type(e).__name__
                }
            })
        raise


def print_comprehensive_results(result: Dict[str, Any]):
    """Print detailed results with comprehensive analysis."""
    
    print(f"📊 Total Iterations: {result['total_iterations']}")
    print(f"🏆 Best Iteration: {result['best_iteration']['iteration']} (score: {result['best_iteration']['score']:.3f})")
    
    # Performance evolution
    print("\n📈 Performance Evolution:")
    history = result["history"]
    for i, scores in enumerate(history["scores"], 1):
        if isinstance(scores, dict):
            avg_score = sum(scores.values()) / len(scores)
            print(f"   Iteration {i}: avg={avg_score:.3f} | {scores}")
    
    # Trend analysis
    print("\n📊 Trend Analysis:")
    trends = result.get("trend_analysis", {})
    if trends.get("trend_available"):
        for metric, trend_data in trends.items():
            if metric.endswith("_trends") and isinstance(trend_data, dict):
                metric_name = metric.replace("_trends", "")
                improvement = trend_data.get("total_improvement", 0)
                print(f"   {metric_name}: improvement={improvement:.3f}, best={trend_data.get('best_score', 0):.3f}")
    
    # Final output preview
    print("\n📝 Final Output Preview:")
    if history.get("outputs"):
        final_output = history["outputs"][-1]
        preview = final_output[:200] + "..." if len(final_output) > 200 else final_output
        print(f"   {preview}")


def analyze_experiment_insights(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key insights from the experiment results."""
    
    insights = {
        "convergence_achieved": False,
        "improvement_detected": False,
        "consistency_high": False,
        "quality_metrics": {}
    }
    
    history = result["history"]
    if len(history.get("scores", [])) >= 2:
        final_scores = history["scores"][-1]
        first_scores = history["scores"][0]
        
        # Check for improvements
        if isinstance(final_scores, dict) and isinstance(first_scores, dict):
            for metric in final_scores:
                if metric in first_scores:
                    improvement = final_scores[metric] - first_scores[metric]
                    insights["quality_metrics"][f"{metric}_improvement"] = improvement
                    if improvement > 0.1:  # Significant improvement threshold
                        insights["improvement_detected"] = True
        
        # Check convergence indicators
        if final_scores.get("convergence", 0) > 0.8:
            insights["convergence_achieved"] = True
        
        if final_scores.get("consistency", 0) > 0.9:
            insights["consistency_high"] = True
    
    return insights


# Custom metrics for domain-specific tracking
@weave.op()
def track_scientific_terminology_usage(text: str) -> Dict[str, Any]:
    """Track usage of scientific terminology across outputs."""
    
    scientific_domains = {
        "quantum": ["quantum", "qubit", "superposition", "entanglement", "decoherence"],
        "computational": ["algorithm", "simulation", "computational", "optimization", "processing"],
        "chemistry": ["molecular", "protein", "drug", "chemical", "pharmaceutical"],
        "research": ["research", "study", "experiment", "analysis", "investigation"]
    }
    
    text_lower = text.lower()
    domain_scores = {}
    
    for domain, terms in scientific_domains.items():
        count = sum(1 for term in terms if term in text_lower)
        domain_scores[f"{domain}_terminology"] = count / len(terms)
    
    # Log terminology analysis to call summary
    call = weave.get_current_call()
    if call and call.summary:
        call.summary.update({
            "terminology_analysis": {
                "text_length": len(text),
                "domain_scores": domain_scores,
                "total_scientific_terms": sum(domain_scores.values())
            }
        })
    
    return domain_scores


if __name__ == "__main__":
    print("🧪 Comprehensive Weave Tracking Demo for Science Workflows")
    print("=" * 70)
    print("This demo showcases comprehensive tracking across:")
    print("  📡 Model Generation (already tracked in generation.py)")
    print("  🔄 Workflow Orchestration")
    print("  🎯 Oracle Evaluations (single & multi-round)")
    print("  📝 Prompt Construction & History")
    print("  📊 Trend Analysis & Performance Evolution")
    print("  🧠 Custom Scientific Metrics")
    print("\n🔗 Check your Weave dashboard for comprehensive tracking data!")
    print("=" * 70)
    
    # Run the comprehensive example
    asyncio.run(run_comprehensive_tracking_example())
    
    print("\n✨ Demo complete! Check your Weave dashboard for:")
    print("  • Complete workflow execution trace")
    print("  • Individual iteration details")
    print("  • Model performance metrics")
    print("  • Evaluation score evolution")
    print("  • Prompt construction logs")
    print("  • Trend analysis results")
    print("  • Custom scientific insights") 