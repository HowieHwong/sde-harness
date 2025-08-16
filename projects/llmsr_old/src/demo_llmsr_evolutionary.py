#!/usr/bin/env python3
"""Demonstration of LLMSR evolutionary equation discovery using original LLMSR framework."""

import json
import time
from pathlib import Path
from typing import Dict, Any

from workflow_llmsr_evolutionary import LLMSROriginalEvolutionaryWorkflow
from workflow import LLMSRWorkflowSync


def demo_single_problem():
    """Demonstrate LLMSR evolutionary equation discovery on a single problem."""
    print("="*80)
    print("LLMSR EVOLUTIONARY EQUATION DISCOVERY DEMO - SINGLE PROBLEM")
    print("="*80)
    
    # Initialize LLMSR evolutionary workflow
    print("Initializing LLMSR evolutionary workflow...")
    llmsr_evolutionary_workflow = LLMSROriginalEvolutionaryWorkflow(
        model_name="openai/gpt-4o-2024-08-06",
        max_iterations=5,
        num_islands=4,
        functions_per_prompt=3,
        cluster_sampling_temperature_init=1.0,
        cluster_sampling_temperature_period=100,
        reset_period=100
    )
    
    # Setup dataset
    print("Setting up dataset...")
    llmsr_evolutionary_workflow.setup_dataset("lsrtransform")
    
    # Get available problems
    problems = [p.name for p in llmsr_evolutionary_workflow.dataset_loader.problems]
    print(f"Available problems: {problems[:5]}... (total: {len(problems)})")
    
    # Choose a problem to demonstrate
    problem_name = "kepler_1"  # You can change this to any available problem
    
    print(f"\nRunning LLMSR evolutionary discovery for problem: {problem_name}")
    print("-" * 60)
    
    # Run LLMSR evolutionary discovery
    start_time = time.time()
    result = llmsr_evolutionary_workflow.discover_equation_llmsr_evolutionary(
        problem_name, 
        output_dir="demo_results"
    )
    end_time = time.time()
    
    # Print results
    print(f"\nLLMSR evolutionary discovery completed in {end_time - start_time:.2f} seconds")
    print("-" * 60)
    
    # Get best equation
    best_equation = llmsr_evolutionary_workflow.get_best_equation_llmsr_evolutionary(problem_name)
    if best_equation:
        print(f"Best NMSE: {best_equation['nmse_score']:.6f}")
        print(f"Found in iteration: {best_equation['iteration']}")
        
        # Extract equation body
        from core.generation import LLMSRGeneration
        generator = LLMSRGeneration()
        equation_body = generator.extract_equation_body(best_equation['equation_code'])
        print(f"Equation: {equation_body}")
    
    # Print LLMSR evolutionary statistics
    llmsr_stats = result.get("llmsr_evolutionary_statistics", {})
    print(f"\nLLMSR Evolutionary Statistics:")
    print(f"  Total programs generated: {llmsr_stats.get('total_programs', 0)}")
    print(f"  Non-empty islands: {llmsr_stats.get('non_empty_islands', 0)}")
    print(f"  Best NMSE history: {llmsr_stats.get('best_nmse_history', float('inf')):.6f}")
    
    # Print island statistics
    island_stats = llmsr_stats.get("island_stats", [])
    for island in island_stats:
        print(f"  Island {island['island_id']}: {island['num_clusters']} clusters, "
              f"best NMSE: {island['best_score']:.6f}")
    
    return result


def demo_comparison():
    """Demonstrate comparison between baseline and LLMSR evolutionary approaches."""
    print("\n" + "="*80)
    print("COMPARISON DEMO - BASELINE vs LLMSR EVOLUTIONARY")
    print("="*80)
    
    # Initialize both workflows
    print("Initializing workflows...")
    baseline_workflow = LLMSRWorkflowSync(
        model_name="openai/gpt-4o-2024-08-06",
        max_iterations=5
    )
    
    llmsr_evolutionary_workflow = LLMSROriginalEvolutionaryWorkflow(
        model_name="openai/gpt-4o-2024-08-06",
        max_iterations=5,
        num_islands=4,
        functions_per_prompt=3,
        cluster_sampling_temperature_init=1.0
    )
    
    # Setup datasets
    baseline_workflow.setup_dataset("lsrtransform")
    llmsr_evolutionary_workflow.setup_dataset("lsrtransform")
    
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
    
    # Run LLMSR evolutionary
    print("Running LLMSR evolutionary approach...")
    llmsr_evolutionary_start = time.time()
    llmsr_evolutionary_result = llmsr_evolutionary_workflow.discover_equation_llmsr_evolutionary(
        problem_name, 
        output_dir="demo_results/llmsr_evolutionary"
    )
    llmsr_evolutionary_end = time.time()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    baseline_best = baseline_workflow.get_best_equation(problem_name)
    llmsr_evolutionary_best = llmsr_evolutionary_workflow.get_best_equation_llmsr_evolutionary(problem_name)
    
    print(f"Baseline:")
    print(f"  Best NMSE: {baseline_best['nmse_score']:.6f}" if baseline_best else "  No valid equation found")
    print(f"  Time: {baseline_end - baseline_start:.2f} seconds")
    
    print(f"\nLLMSR Evolutionary:")
    print(f"  Best NMSE: {llmsr_evolutionary_best['nmse_score']:.6f}" if llmsr_evolutionary_best else "  No valid equation found")
    print(f"  Time: {llmsr_evolutionary_end - llmsr_evolutionary_start:.2f} seconds")
    
    if baseline_best and llmsr_evolutionary_best:
        improvement = baseline_best['nmse_score'] - llmsr_evolutionary_best['nmse_score']
        print(f"\nImprovement: {improvement:.6f} (lower is better)")
        print(f"LLMSR Evolutionary {'better' if improvement > 0 else 'worse' if improvement < 0 else 'same'}")
    
    # Print LLMSR evolutionary statistics
    llmsr_stats = llmsr_evolutionary_result.get("llmsr_evolutionary_statistics", {})
    print(f"\nLLMSR Evolutionary Statistics:")
    print(f"  Total programs: {llmsr_stats.get('total_programs', 0)}")
    print(f"  Non-empty islands: {llmsr_stats.get('non_empty_islands', 0)}")
    print(f"  Best NMSE history: {llmsr_stats.get('best_nmse_history', float('inf')):.6f}")


def demo_parameter_sensitivity():
    """Demonstrate sensitivity to LLMSR evolutionary parameters."""
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
        
        workflow = LLMSROriginalEvolutionaryWorkflow(
            model_name="openai/gpt-4o-2024-08-06",
            max_iterations=3,  # Shorter for demo
            cluster_sampling_temperature_init=temp,
            functions_per_prompt=2
        )
        
        workflow.setup_dataset("lsrtransform")
        
        start_time = time.time()
        result = workflow.discover_equation_llmsr_evolutionary(
            problem_name, 
            output_dir=f"demo_results/temp_{temp}"
        )
        end_time = time.time()
        
        best_equation = workflow.get_best_equation_llmsr_evolutionary(problem_name)
        llmsr_stats = result.get("llmsr_evolutionary_statistics", {})
        
        results[temp] = {
            "nmse": best_equation['nmse_score'] if best_equation else float('inf'),
            "time": end_time - start_time,
            "total_programs": llmsr_stats.get('total_programs', 0),
            "non_empty_islands": llmsr_stats.get('non_empty_islands', 0)
        }
        
        print(f"  NMSE: {results[temp]['nmse']:.6f}")
        print(f"  Time: {results[temp]['time']:.2f}s")
        print(f"  Total programs: {results[temp]['total_programs']}")
        print(f"  Non-empty islands: {results[temp]['non_empty_islands']}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEMPERATURE SENSITIVITY SUMMARY")
    print("="*60)
    
    for temp, result in results.items():
        print(f"Temperature {temp}: NMSE={result['nmse']:.6f}, "
              f"Time={result['time']:.2f}s, "
              f"Programs={result['total_programs']}, "
              f"Islands={result['non_empty_islands']}")


def demo_island_configurations():
    """Demonstrate different island configurations."""
    print("\n" + "="*80)
    print("ISLAND CONFIGURATIONS DEMO")
    print("="*80)
    
    problem_name = "kepler_1"
    configurations = [
        {"num_islands": 2, "functions_per_prompt": 2, "name": "Small"},
        {"num_islands": 4, "functions_per_prompt": 3, "name": "Medium"},
        {"num_islands": 8, "functions_per_prompt": 4, "name": "Large"}
    ]
    results = {}
    
    for config in configurations:
        print(f"\nTesting {config['name']} configuration: {config['num_islands']} islands, {config['functions_per_prompt']} functions per prompt")
        print("-" * 60)
        
        workflow = LLMSROriginalEvolutionaryWorkflow(
            model_name="openai/gpt-4o-2024-08-06",
            max_iterations=3,  # Shorter for demo
            num_islands=config['num_islands'],
            functions_per_prompt=config['functions_per_prompt']
        )
        
        workflow.setup_dataset("lsrtransform")
        
        start_time = time.time()
        result = workflow.discover_equation_llmsr_evolutionary(
            problem_name, 
            output_dir=f"demo_results/config_{config['name'].lower()}"
        )
        end_time = time.time()
        
        best_equation = workflow.get_best_equation_llmsr_evolutionary(problem_name)
        llmsr_stats = result.get("llmsr_evolutionary_statistics", {})
        
        results[config['name']] = {
            "nmse": best_equation['nmse_score'] if best_equation else float('inf'),
            "time": end_time - start_time,
            "total_programs": llmsr_stats.get('total_programs', 0),
            "non_empty_islands": llmsr_stats.get('non_empty_islands', 0),
            "config": config
        }
        
        print(f"  NMSE: {results[config['name']]['nmse']:.6f}")
        print(f"  Time: {results[config['name']]['time']:.2f}s")
        print(f"  Total programs: {results[config['name']]['total_programs']}")
        print(f"  Non-empty islands: {results[config['name']]['non_empty_islands']}")
    
    # Print summary
    print("\n" + "="*60)
    print("ISLAND CONFIGURATIONS SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        config = result['config']
        print(f"{name} ({config['num_islands']} islands, {config['functions_per_prompt']} functions): "
              f"NMSE={result['nmse']:.6f}, "
              f"Time={result['time']:.2f}s, "
              f"Programs={result['total_programs']}, "
              f"Islands={result['non_empty_islands']}")


def main():
    """Run all demonstrations."""
    print("LLMSR Evolutionary Framework Demonstration")
    print("This demo shows the LLMSR evolutionary equation discovery capabilities using the original LLMSR framework.")
    print("Note: This demo requires API credentials and may incur costs.")
    print("Make sure the original LLMSR code is available at ../llmsr/")
    
    # Create output directory
    Path("demo_results").mkdir(exist_ok=True)
    
    try:
        # Run demonstrations
        demo_single_problem()
        demo_comparison()
        demo_parameter_sensitivity()
        demo_island_configurations()
        
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
        print("4. Original LLMSR code available at ../llmsr/")


if __name__ == "__main__":
    main()


