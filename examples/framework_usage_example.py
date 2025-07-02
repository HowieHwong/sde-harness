"""
Framework Usage Example: Proper way to use sci_demo with Weave tracking.

This example demonstrates:
1. How users should initialize weave with their own project name
2. How the framework works with and without weave tracking
3. Best practices for framework usage
"""

import os
import sys
import asyncio

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# STEP 1: User initializes weave with their project name (OPTIONAL)
# Comment this out to see the framework work without tracking
import weave
weave.init("my_science_research_project")  # USER controls the project name!

# STEP 2: Import framework components AFTER weave initialization  
from sci_demo import Generation, Workflow, Oracle, Prompt
from sci_demo.oracle import improvement_rate_metric, consistency_metric

print("🧪 Framework Usage Example")
print("=" * 50)

# Check if weave tracking is enabled
from sci_demo import is_weave_initialized
if is_weave_initialized():
    print("✅ Weave tracking is ENABLED - check your dashboard!")
    print(f"   Project: my_science_research_project")
else:
    print("⚪ Weave tracking is DISABLED - running without tracking")

print("=" * 50)

def setup_science_framework():
    """Setup the science framework components."""
    
    # Initialize Generation component
    gen = Generation(
        models_file="models.yaml",
        credentials_file="credentials.yaml",
        max_workers=2
    )
    
    # Setup Oracle with domain-specific metrics
    oracle = Oracle()
    
    # Register domain-specific metrics
    def scientific_accuracy(pred, ref, **kwargs):
        """Domain-specific accuracy metric."""
        pred_terms = set(pred.lower().split())
        ref_terms = set(ref.lower().split()) 
        if not ref_terms:
            return 1.0
        return len(pred_terms.intersection(ref_terms)) / len(ref_terms)
    
    def clarity_score(pred, ref, **kwargs):
        """Measure clarity of scientific writing."""
        sentences = pred.split('.')
        if not sentences:
            return 0.0
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        # Optimal: 15-25 words per sentence for scientific writing
        if 15 <= avg_words <= 25:
            return 1.0
        elif avg_words < 15:
            return avg_words / 15
        else:
            return 25 / avg_words
    
    oracle.register_metric("scientific_accuracy", scientific_accuracy)
    oracle.register_metric("clarity", clarity_score)
    
    # Register multi-round metrics for iterative improvement
    oracle.register_multi_round_metric("improvement_rate", improvement_rate_metric)
    oracle.register_multi_round_metric("consistency", consistency_metric)
    
    # Create workflow with framework settings
    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=3,
        enable_history_in_prompts=True,
        enable_multi_round_metrics=True
    )
    
    return gen, oracle, workflow

async def run_science_experiment():
    """Run a science experiment using the framework."""
    
    # Setup framework
    gen, oracle, workflow = setup_science_framework()
    
    # Create a scientific prompt
    prompt = Prompt(
        template_name="iterative_with_feedback",
        default_vars={
            "task_description": "Explain quantum computing applications in drug discovery",
            "input_text": "How can quantum computers help find new medicines?",
            "additional_instructions": "Focus on specific mechanisms and real-world examples."
        }
    )
    
    # Define reference for evaluation
    reference = """
    Quantum computers can accelerate drug discovery through molecular simulation, 
    protein folding prediction, and optimization of drug properties. They excel 
    at modeling quantum mechanical effects in chemical reactions that classical 
    computers struggle with.
    """
    
    # User-defined experiment metadata (tracked if weave is enabled)
    # Note: Individual experiment metadata can be added via attributes or call metadata
    if is_weave_initialized():
        print("💡 Experiment metadata can be added via weave.attributes() or call parameters")
    
    print("🔬 Running science experiment...")
    
    # Run the experiment (automatically tracked if weave is enabled)
    result = await workflow.run(
        prompt=prompt,
        reference=reference,
        gen_args={
            "model_name": "gpt-4o",  # Adjust based on your models.yaml
            "max_tokens": 200,
            "temperature": 0.7
        },
        history_context={
            "experiment_id": "quantum_drug_discovery_001",
            "domain": "computational_chemistry"
        }
    )
    
    print("✅ Experiment completed!")
    print(f"   Total iterations: {result['total_iterations']}")
    print(f"   Final scores: {result['final_scores']}")
    print(f"   Best iteration: {result['best_iteration']['iteration']}")
    
    # Show final output
    if result['history']['outputs']:
        final_output = result['history']['outputs'][-1]
        print(f"\n📝 Final Output Preview:")
        print(f"   {final_output[:150]}...")
    
    return result

def demonstrate_framework_flexibility():
    """Show how the framework works with different configurations."""
    
    print("\n🔧 Framework Flexibility Demo")
    print("-" * 30)
    
    # Example 1: Without weave (tracking disabled)
    print("1. Framework works without weave tracking")
    
    # Example 2: Custom metrics
    oracle = Oracle()
    oracle.register_metric("custom_metric", lambda p, r: 0.85)
    print("2. Easy custom metric registration")
    
    # Example 3: Different prompt templates
    templates = ["iterative", "iterative_with_feedback", "conversation"]
    for template in templates:
        prompt = Prompt(template_name=template, default_vars={"input_text": "test"})
        print(f"3. Template '{template}': {len(prompt.build())} chars")

if __name__ == "__main__":
    print("🚀 Starting Framework Usage Example")
    print("This shows the PROPER way to use sci_demo as a framework\n")
    
    # Run the main experiment
    try:
        result = asyncio.run(run_science_experiment())
        
        # Show framework flexibility
        demonstrate_framework_flexibility()
        
        print("\n🎯 Key Framework Principles:")
        print("  1. USER calls weave.init() with THEIR project name")
        print("  2. Framework imports come AFTER weave initialization")
        print("  3. Tracking is OPTIONAL and controlled by the user")
        print("  4. Framework works with or without weave")
        print("  5. All @weave.op() decorators are still present for tracking")
        
        if is_weave_initialized():
            print("\n📊 Check your Weave dashboard for comprehensive tracking!")
        else:
            print("\n⚪ To enable tracking, uncomment the weave.init() line at the top")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your models.yaml and credentials.yaml are configured") 