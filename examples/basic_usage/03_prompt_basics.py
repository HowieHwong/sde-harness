#!/usr/bin/env python3
"""
Prompt Basics Example

This example demonstrates the core functionality of the Prompt class,
including template usage, variable substitution, and dynamic prompt building.

Run this example:
    python examples/basic_usage/03_prompt_basics.py
"""

import sys
import os
from typing import Dict, Any, List

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from sde_harness.core import Prompt


def example_builtin_templates():
    """Example 1: Using built-in templates"""
    print("🔷 Example 1: Built-in Templates")
    print("-" * 50)

    # Available built-in templates
    builtin_templates = ["summarize", "translate", "qa", "few_shot", "iterative"]

    for template_name in builtin_templates:
        try:
            prompt = Prompt(template_name=template_name)
            print(f"\n📝 Template: {template_name}")
            print(f"Raw template:\n{prompt.template}")
            print("-" * 30)

            # Show variables in template (extract from template string)
            import re

            variables = re.findall(r"\{(.*?)\}", prompt.template)
            if variables:
                print(f"Variables found: {variables}")
            else:
                print("No variables found in template")

        except Exception as e:
            print(f"❌ Error loading template '{template_name}': {e}")

    print()


def example_template_with_variables():
    """Example 2: Using templates with variable substitution"""
    print("🔷 Example 2: Template with Variables")
    print("-" * 50)

    # Use the summarize template
    prompt = Prompt(template_name="summarize")

    print("Original template:")
    print(prompt.template)
    print()

    # Set variables and build prompt
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from and make predictions or decisions based on data. It has applications 
    in many fields including computer vision, natural language processing, and robotics.
    """

    built_prompt = prompt.build(
        {"input_text": sample_text.strip(), "max_length": "100 words"}
    )

    print("Built prompt with variables:")
    print(built_prompt)
    print()


def example_custom_template():
    """Example 3: Creating custom templates"""
    print("🔷 Example 3: Custom Templates")
    print("-" * 50)

    # Define a custom template for scientific analysis
    custom_template = """
You are a scientific research assistant. Analyze the following {data_type} data and provide insights.

Data Description: {description}
Data: {data}

Please provide:
1. Key observations
2. Potential patterns or trends
3. Suggested next steps for investigation
4. Confidence level: {confidence_level}

Analysis:"""

    prompt = Prompt(custom_template=custom_template)

    print("Custom template:")
    print(prompt.template)
    print()

    # Build with variables
    built_prompt = prompt.build(
        {
            "data_type": "experimental",
            "description": "Temperature measurements over time",
            "data": "Day 1: 22°C, Day 2: 24°C, Day 3: 26°C, Day 4: 25°C, Day 5: 23°C",
            "confidence_level": "high",
        }
    )

    print("Built custom prompt:")
    print(built_prompt)
    print()


def example_default_variables():
    """Example 4: Using default variables"""
    print("🔷 Example 4: Default Variables")
    print("-" * 50)

    # Template with default variables
    template = """
Research Question: {question}
Domain: {domain}
Method: {method}
Expected Outcome: {expected_outcome}

Please provide a detailed research plan.
"""

    # Create prompt with default variables
    prompt = Prompt(
        custom_template=template,
        default_vars={
            "domain": "Computer Science",
            "method": "Experimental Study",
            "expected_outcome": "Improved performance",
        },
    )

    print("Template with defaults:")
    print(prompt.template)
    print()

    # Build with only required variable
    built_prompt1 = prompt.build(
        {"question": "How can we improve machine learning model accuracy?"}
    )

    print("Built with defaults:")
    print(built_prompt1)
    print("-" * 30)

    # Build with overridden defaults
    built_prompt2 = prompt.build(
        {
            "question": "What is the impact of quantum computing on cryptography?",
            "domain": "Quantum Computing",
            "method": "Theoretical Analysis",
            "expected_outcome": "Security implications identified",
        }
    )

    print("Built with overridden defaults:")
    print(built_prompt2)
    print()


def example_dynamic_variables():
    """Example 5: Dynamic variable management"""
    print("🔷 Example 5: Dynamic Variable Management")
    print("-" * 50)

    template = """
Experiment: {experiment_name}
Hypothesis: {hypothesis}
Variables: {variables}
Control Group: {control_group}
Test Group: {test_group}

Analysis: {analysis}
"""

    prompt = Prompt(custom_template=template)

    # Add variables dynamically
    prompt.add_vars(
        experiment_name="ML Model Comparison",
        hypothesis="Model A will outperform Model B",
        variables="Training data, model architecture, hyperparameters",
    )

    print("After adding some variables:")
    print(f"Current variables: {list(prompt.default_vars.keys())}")
    print()

    # Build with additional variables
    built_prompt = prompt.build(
        {
            "control_group": "Baseline model with default parameters",
            "test_group": "Optimized model with tuned parameters",
            "analysis": "Statistical significance testing will be performed",
        }
    )

    print("Built prompt:")
    print(built_prompt)
    print()

    # Clear and reset variables
    prompt.default_vars.clear()
    prompt.add_vars(
        experiment_name="Neural Network Architecture Study",
        hypothesis="Deeper networks improve accuracy",
        variables="Network depth, width, activation functions",
    )

    print("After clearing and resetting:")
    print(f"New variables: {list(prompt.default_vars.keys())}")
    print()


def example_iterative_prompts():
    """Example 6: Iterative prompt building (simulating multi-round)"""
    print("🔷 Example 6: Iterative Prompt Building")
    print("-" * 50)

    # Base template for iterative research
    base_template = """
Research Context: {context}
Previous Findings: {previous_findings}
Current Focus: {current_focus}

Please provide the next research direction based on the above information.

Response:"""

    prompt = Prompt(
        custom_template=base_template,
        default_vars={
            "context": "Investigating machine learning optimization techniques",
            "previous_findings": "None (first iteration)",
        },
    )

    # Simulate multiple iterations
    iterations = [
        {
            "current_focus": "Initial survey of optimization algorithms",
            "previous_findings": "None (first iteration)",
        },
        {
            "current_focus": "Comparing gradient descent variants",
            "previous_findings": "Adam optimizer shows promise for deep networks",
        },
        {
            "current_focus": "Investigating adaptive learning rates",
            "previous_findings": "Adam optimizer shows promise; need to study learning rate schedules",
        },
    ]

    for i, iteration_vars in enumerate(iterations, 1):
        print(f"Iteration {i}:")

        # Update variables for this iteration
        prompt.add_vars(**iteration_vars)

        # Build the prompt
        built_prompt = prompt.build()

        print(built_prompt)
        print("-" * 40)

    print()


def example_complex_template():
    """Example 7: Complex template with conditional content"""
    print("🔷 Example 7: Complex Template Usage")
    print("-" * 50)

    # Complex template for experimental design
    complex_template = """
=== EXPERIMENTAL DESIGN TEMPLATE ===

Project: {project_name}
Researcher: {researcher}
Date: {date}

OBJECTIVE:
{objective}

METHODOLOGY:
Primary Method: {primary_method}
Secondary Methods: {secondary_methods}

DATA COLLECTION:
Sample Size: {sample_size}
Data Sources: {data_sources}
Collection Period: {collection_period}

ANALYSIS PLAN:
Statistical Methods: {statistical_methods}
Success Metrics: {success_metrics}
Expected Results: {expected_results}

RESOURCES REQUIRED:
Budget: {budget}
Equipment: {equipment}
Personnel: {personnel}

TIMELINE:
Phase 1: {phase_1}
Phase 2: {phase_2}
Phase 3: {phase_3}

Please review this experimental design and provide feedback on:
1. Methodology appropriateness
2. Potential limitations
3. Suggested improvements
4. Risk assessment
"""

    prompt = Prompt(
        custom_template=complex_template,
        default_vars={
            "researcher": "Research Team",
            "date": "2024",
            "secondary_methods": "Literature review, expert interviews",
            "statistical_methods": "Descriptive statistics, hypothesis testing",
            "budget": "To be determined",
            "equipment": "Standard laboratory equipment",
            "personnel": "2-3 researchers",
        },
    )

    # Build a complete experimental design
    built_prompt = prompt.build(
        {
            "project_name": "Impact of AI on Scientific Discovery",
            "objective": "Evaluate how AI tools enhance research productivity and quality",
            "primary_method": "Controlled experiment with AI-assisted vs traditional research",
            "sample_size": "100 researchers across 5 institutions",
            "data_sources": "Survey responses, productivity metrics, publication quality scores",
            "collection_period": "6 months",
            "success_metrics": "Time to completion, research quality scores, researcher satisfaction",
            "expected_results": "20-30% improvement in research efficiency",
            "phase_1": "Baseline data collection (2 months)",
            "phase_2": "AI tool deployment and training (2 months)",
            "phase_3": "Data collection and analysis (2 months)",
        }
    )

    print(built_prompt)
    print()


def example_error_handling():
    """Example 8: Error handling in prompts"""
    print("🔷 Example 8: Error Handling")
    print("-" * 50)

    # Template with required variable
    template = "Research question: {question}\nMethod: {method}\nAnalysis: {analysis}"
    prompt = Prompt(custom_template=template)

    # Test missing variable
    try:
        built_prompt = prompt.build(
            {"question": "What is AI?"}
        )  # Missing 'method' and 'analysis'
        print(f"Unexpected success: {built_prompt}")
    except Exception as e:
        print(f"✅ Expected error for missing variables: {type(e).__name__}: {e}")

    # Test with all variables
    try:
        built_prompt = prompt.build(
            {
                "question": "What is AI?",
                "method": "Literature review",
                "analysis": "Systematic analysis of definitions",
            }
        )
        print(f"✅ Success with all variables: {built_prompt[:100]}...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # Test invalid template
    try:
        bad_prompt = Prompt(custom_template="Invalid template with {unclosed_var")
        built = bad_prompt.build()
        print(f"Unexpected success with bad template: {built}")
    except Exception as e:
        print(f"✅ Expected error for invalid template: {type(e).__name__}")

    print()


def main():
    """Run all Prompt examples"""
    print("🚀 SDE-Harness Prompt Examples")
    print("=" * 60)
    print()

    example_builtin_templates()
    example_template_with_variables()
    example_custom_template()
    example_default_variables()
    example_dynamic_variables()
    example_iterative_prompts()
    example_complex_template()
    example_error_handling()

    print("✅ All Prompt examples completed!")
    print("\n💡 Next Steps:")
    print("- Try examples/basic_usage/04_workflow_basics.py")
    print("- Create domain-specific prompt templates")
    print("- Experiment with dynamic prompt generation")
    print("- Combine prompts with Generation and Oracle classes")


if __name__ == "__main__":
    main()
