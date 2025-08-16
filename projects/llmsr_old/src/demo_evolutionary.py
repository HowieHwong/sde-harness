#!/usr/bin/env python3
"""Demonstration of evolutionary equation discovery using LLMSR."""

import json
import time
from pathlib import Path
from typing import Dict, Any

from workflow_evolutionary import LLMSREvolutionaryWorkflow
from workflow import LLMSRWorkflowSync


def demo_single_problem():
    """Demonstrate evolutionary equation discovery on a single problem."""
    print("="*80)
    print("EVOLUTIONARY EQUATION DISCOVERY DEMO - SINGLE PROBLEM")
    print("="*80)
    
    # Initialize evolutionary workflow
    print("Initializing evolutionary workflow...")
    evolutionary_workflow = LLMSREvolutionaryWorkflow(
        model_name="openai/gpt-4o-2024-08-06",
        max_iterations=5,
        num_islands=4,
        max_programs_per_island=50,
        num_examples=3,
        temperature=1.0,
        sampling_strategy="boltzmann"
    )
    
    # Setup dataset
    print("Setting up dataset...")
    evolutionary_workflow.setup_dataset("lsrtransform")
    
    # Get available problems
    problems = [p.name for p in evolutionary_workflow.dataset_loader.problems]
    print(f"Available problems: {problems[:5]}... (total: {len(problems)})")
    
    # Choose a problem to demonstrate
    problem_name = "kepler_1"  # You can change this to any available problem
    
    print(f"\nRunning evolutionary discovery for problem: {problem_name}")
    print("-" * 60)
    
    # Run evolutionary discovery
    start_time = time.time()
    result = evolutionary_workflow.discover_equation_evolutionary(
        problem_name, 
        output_dir="demo_results"
    )
    end_time = time.time()
    
    # Print results
    print(f"\nEvolutionary discovery completed in {end_time - start_time:.2f} seconds")
    print("-" * 60)
    
    # Get best equation
    best_equation = evolutionary_workflow.get_best_equation_evolutionary(problem_name)
    if best_equation:
        print(f"Best NMSE: {best_equation['nmse_score']:.6f}")
        print(f"Found in iteration: {best_equation['iteration']}")
        
        # Extract equation body
        from core.generation import LLMSRGeneration
        generator = LLMSRGeneration()
        equation_body = generator.extract_equation_body(best_equation['equation_code'])
        print(f"Equation: {equation_body}")
    
    # Print evolutionary statistics
    evo_stats = result.get("evolutionary_statistics", {})
    print(f"\nEvolutionary Statistics:")
    print(f"  Total programs generated: {evo_stats.get('total_programs', 0)}")
    print(f"  Programs that improved: {evo_stats.get('programs_improved', 0)}")
    print(f"  Improvement rate: {evo_stats.get('improvement_rate', 0):.3f}")
    print(f"  Non-empty islands: {evo_stats.get('non_empty_islands', 0)}")
    
    return result


def demo_comparison():
    """Demonstrate comparison between baseline and evolutionary approaches."""
    print("\n" + "="*80)
    print("COMPARISON DEMO - BASELINE vs EVOLUTIONARY")
    print("="*80)
    
    # Initialize both workflows
    print("Initializing workflows...")
    baseline_workflow = LLMSRWorkflowSync(
        model_name="openai/gpt-4o-2024-08-06",
        max_iterations=5
    )
    
    evolutionary_workflow = LLMSREvolutionaryWorkflow(
        model_name="openai/gpt-4o-2024-08-06",
        max_iterations=5,
        num_islands=4,
        num_examples=3,
        temperature=1.0
    )
    
    # Setup datasets
    baseline_workflow.setup_dataset("lsrtransform")
    evolutionary_workflow.setup_dataset("lsrtransform")
    
    # Choose a problem for comparison
    problem_name = "kepler_1"
    
    print(f"\nRunning comparison for problem: {problem_name}")
    print("-" * 60)
    
    # Run baseline
    print("Running baseline approach...")
    baseline_start = time.time()
    baseline_result = baseline_workflow.discover_equation_sync(
        problem_name, 
        output_dir="demo_results/baseline"
    )
    baseline_end = time.time()
    
    # Run evolutionary
    print("Running evolutionary approach...")
    evolutionary_start = time.time()
    evolutionary_result = evolutionary_workflow.discover_equation_evolutionary(
        problem_name, 
        output_dir="demo_results/evolutionary"
    )
    evolutionary_end = time.time()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    baseline_best = baseline_workflow.get_best_equation(problem_name)
    evolutionary_best = evolutionary_workflow.get_best_equation_evolutionary(problem_name)
    
    print(f"Baseline:")
    print(f"  Best NMSE: {baseline_best['nmse_score']:.6f}" if baseline_best else "  No valid equation found")
    print(f"  Time: {baseline_end - baseline_start:.2f} seconds")
    
    print(f"\nEvolutionary:")
    print(f"  Best NMSE: {evolutionary_best['nmse_score']:.6f}" if evolutionary_best else "  No valid equation found")
    print(f"  Time: {evolutionary_end - evolutionary_start:.2f} seconds")
    
    if baseline_best and evolutionary_best:
        improvement = baseline_best['nmse_score'] - evolutionary_best['nmse_score']
        print(f"\nImprovement: {improvement:.6f} (lower is better)")
        print(f"Evolutionary {'better' if improvement > 0 else 'worse' if improvement < 0 else 'same'}")
    
    # Print evolutionary statistics
    evo_stats = evolutionary_result.get("evolutionary_statistics", {})
    print(f"\nEvolutionary Statistics:")
    print(f"  Total programs: {evo_stats.get('total_programs', 0)}")
    print(f"  Improvements: {evo_stats.get('programs_improved', 0)}")
    print(f"  Improvement rate: {evo_stats.get('improvement_rate', 0):.3f}")


def demo_parameter_sensitivity():
    """Demonstrate sensitivity to evolutionary parameters."""
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY DEMO")
    print("="*80)
    
    problem_name = "kepler_1"
    
    # Test different temperatures
    temperatures = [0.5, 1.0, 2.0]
    results = {}
    
    for temp in temperatures:
        print(f"\nTesting temperature: {temp}")
        print("-" * 40)
        
        workflow = LLMSREvolutionaryWorkflow(
            model_name="openai/gpt-4o-2024-08-06",
            max_iterations=3,  # Shorter for demo
            temperature=temp,
            num_examples=2
        )
        
        workflow.setup_dataset("lsrtransform")
        
        start_time = time.time()
        result = workflow.discover_equation_evolutionary(
            problem_name, 
            output_dir=f"demo_results/temp_{temp}"
        )
        end_time = time.time()
        
        best_equation = workflow.get_best_equation_evolutionary(problem_name)
        evo_stats = result.get("evolutionary_statistics", {})
        
        results[temp] = {
            "nmse": best_equation['nmse_score'] if best_equation else float('inf'),
            "time": end_time - start_time,
            "improvement_rate": evo_stats.get('improvement_rate', 0),
            "total_programs": evo_stats.get('total_programs', 0)
        }
        
        print(f"  NMSE: {results[temp]['nmse']:.6f}")
        print(f"  Time: {results[temp]['time']:.2f}s")
        print(f"  Improvement rate: {results[temp]['improvement_rate']:.3f}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEMPERATURE SENSITIVITY SUMMARY")
    print("="*60)
    
    for temp, result in results.items():
        print(f"Temperature {temp}: NMSE={result['nmse']:.6f}, "
              f"Time={result['time']:.2f}s, "
              f"Improvement={result['improvement_rate']:.3f}")


def demo_sampling_strategies():
    """Demonstrate different sampling strategies."""
    print("\n" + "="*80)
    print("SAMPLING STRATEGIES DEMO")
    print("="*80)
    
    problem_name = "kepler_1"
    strategies = ["boltzmann", "best", "random"]
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        print("-" * 40)
        
        workflow = LLMSREvolutionaryWorkflow(
            model_name="openai/gpt-4o-2024-08-06",
            max_iterations=3,  # Shorter for demo
            sampling_strategy=strategy,
            num_examples=2
        )
        
        workflow.setup_dataset("lsrtransform")
        
        start_time = time.time()
        result = workflow.discover_equation_evolutionary(
            problem_name, 
            output_dir=f"demo_results/strategy_{strategy}"
        )
        end_time = time.time()
        
        best_equation = workflow.get_best_equation_evolutionary(problem_name)
        evo_stats = result.get("evolutionary_statistics", {})
        
        results[strategy] = {
            "nmse": best_equation['nmse_score'] if best_equation else float('inf'),
            "time": end_time - start_time,
            "improvement_rate": evo_stats.get('improvement_rate', 0),
            "total_programs": evo_stats.get('total_programs', 0)
        }
        
        print(f"  NMSE: {results[strategy]['nmse']:.6f}")
        print(f"  Time: {results[strategy]['time']:.2f}s")
        print(f"  Improvement rate: {results[strategy]['improvement_rate']:.3f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SAMPLING STRATEGIES SUMMARY")
    print("="*60)
    
    for strategy, result in results.items():
        print(f"{strategy.capitalize()}: NMSE={result['nmse']:.6f}, "
              f"Time={result['time']:.2f}s, "
              f"Improvement={result['improvement_rate']:.3f}")


def main():
    """Run all demonstrations."""
    print("LLMSR Evolutionary Framework Demonstration")
    print("This demo shows the evolutionary equation discovery capabilities.")
    print("Note: This demo requires API credentials and may incur costs.")
    
    # Create output directory
    Path("demo_results").mkdir(exist_ok=True)
    
    try:
        # Run demonstrations
        demo_single_problem()
        demo_comparison()
        demo_parameter_sensitivity()
        demo_sampling_strategies()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED")
        print("="*80)
        print("Results saved in 'demo_results' directory")
        print("Check the individual result files for detailed information")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Make sure you have:")
        print("1. Valid API credentials in credentials.yaml")
        print("2. Required dependencies installed")
        print("3. Sufficient API credits")


if __name__ == "__main__":
    main()


