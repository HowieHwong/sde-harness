#!/usr/bin/env python3
"""CLI for evolutionary equation discovery using LLMSR."""

import argparse
import json
import sys
import os
import weave
from pathlib import Path
from typing import List, Optional

from workflow_evolutionary import LLMSREvolutionaryWorkflow




def main():
    """Main CLI function for evolutionary equation discovery."""
    parser = argparse.ArgumentParser(
        description="Evolutionary equation discovery using LLMSR",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Problem selection
    problem_group = parser.add_mutually_exclusive_group(required=True)
    problem_group.add_argument(
        "--problem", 
        type=str, 
        help="Single problem name to process"
    )
    problem_group.add_argument(
        "--problems", 
        nargs="+", 
        help="Multiple problem names to process"
    )
    problem_group.add_argument(
        "--all-problems", 
        action="store_true", 
        help="Process all available problems"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evolutionary_results",
        help="Output directory for results (default: evolutionary_results)"
    )
    
    # Model and generation options
    parser.add_argument(
        "--model", 
        type=str, 
        default="openai/gpt-4o-2024-08-06",
        help="LLM model to use (default: openai/gpt-4o-2024-08-06)"
    )
    parser.add_argument(
        "--max-iterations", 
        type=int, 
        default=5,
        help="Maximum number of iterations (default: 5)"
    )
    parser.add_argument(
        "--max-params", 
        type=int, 
        default=10,
        help="Maximum number of parameters to optimize (default: 10)"
    )
    
    # Evolutionary parameters
    parser.add_argument(
        "--num-islands", 
        type=int, 
        default=4,
        help="Number of islands in evolutionary buffer (default: 4)"
    )
    parser.add_argument(
        "--max-programs-per-island", 
        type=int, 
        default=50,
        help="Maximum programs per island (default: 50)"
    )
    parser.add_argument(
        "--reset-period", 
        type=int, 
        default=100,
        help="How often to reset weaker islands (default: 100)"
    )
    parser.add_argument(
        "--reset-fraction", 
        type=float, 
        default=0.5,
        help="Fraction of islands to reset (default: 0.5)"
    )
    parser.add_argument(
        "--num-examples", 
        type=int, 
        default=3,
        help="Number of examples to include in prompts (default: 3)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Temperature for Boltzmann sampling (default: 1.0)"
    )
    parser.add_argument(
        "--sampling-strategy", 
        type=str, 
        choices=["boltzmann", "best", "random"],
        default="boltzmann",
        help="Sampling strategy for examples (default: boltzmann)"
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="lsrtransform",
        help="Dataset name to use (default: lsrtransform)"
    )
    
    # Comparison options
    parser.add_argument(
        "--compare-baseline", 
        type=str, 
        help="Path to baseline results JSON file for comparison"
    )
    
    # Analysis options
    parser.add_argument(
        "--print-summary", 
        action="store_true",
        help="Print detailed summary after completion"
    )
    parser.add_argument(
        "--save-statistics", 
        action="store_true",
        help="Save evolutionary statistics to separate file"
    )

    parser.add_argument("--project-name", 
            default="llm-sym-reg-evol",
            help="Weave project name for logging")
    
    args = parser.parse_args()
    
    # Initialize weave logging
    weave.init(args.project_name)
    
    try:
        # Initialize evolutionary workflow
        print("Initializing evolutionary LLMSR workflow...")
        workflow = LLMSREvolutionaryWorkflow(
            model_name=args.model,
            max_iterations=args.max_iterations,
            max_params=args.max_params,
            num_islands=args.num_islands,
            max_programs_per_island=args.max_programs_per_island,
            reset_period=args.reset_period,
            reset_fraction=args.reset_fraction,
            num_examples=args.num_examples,
            temperature=args.temperature,
            sampling_strategy=args.sampling_strategy
        )
        
        # Setup dataset
        print(f"Setting up dataset: {args.dataset}")
        workflow.setup_dataset(args.dataset)
        
        # Determine problems to process
        if args.problem:
            problems = [args.problem]
        elif args.problems:
            problems = args.problems
        else:  # all_problems
            problems = [p.name for p in workflow.dataset_loader.problems]
        
        print(f"Processing {len(problems)} problem(s): {problems}")
        
        # Process problems
        if len(problems) == 1:
            # Single problem
            result = workflow.discover_equation_evolutionary(problems[0], args.output_dir)
            print(f"\nCompleted evolutionary discovery for {problems[0]}")
            
            # Print best result
            best_equation = workflow.get_best_equation_evolutionary(problems[0])
            if best_equation:
                print(f"Best NMSE: {best_equation['nmse_score']:.6f}")
                print(f"Found in iteration: {best_equation['iteration']}")
            
        else:
            # Multiple problems
            all_results = workflow.discover_all_equations_evolutionary(
                args.output_dir, 
                problem_filter=problems
            )
            print(f"\nCompleted evolutionary discovery for {len(problems)} problems")
        
        # Print summary if requested
        if args.print_summary:
            workflow.print_evolutionary_summary()
        
        # Save statistics if requested
        if args.save_statistics:
            stats = workflow.get_evolutionary_statistics()
            stats_file = Path(args.output_dir) / "evolutionary_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"Evolutionary statistics saved to: {stats_file}")
        
        # Compare with baseline if provided
        if args.compare_baseline:
            baseline_file = Path(args.compare_baseline)
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline_results = json.load(f)
                
                comparison = workflow.compare_with_baseline(baseline_results)
                
                print("\n" + "="*60)
                print("COMPARISON WITH BASELINE")
                print("="*60)
                print(f"Problems compared: {comparison['problems_compared']}")
                print(f"Evolutionary better: {comparison['evolutionary_better']}")
                print(f"Baseline better: {comparison['baseline_better']}")
                print(f"Same performance: {comparison['same_performance']}")
                
                # Save comparison
                comparison_file = Path(args.output_dir) / "baseline_comparison.json"
                with open(comparison_file, 'w') as f:
                    json.dump(comparison, f, indent=2, default=str)
                print(f"Comparison saved to: {comparison_file}")
            else:
                print(f"Warning: Baseline file {args.compare_baseline} not found")
        
        print(f"\nEvolutionary equation discovery completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
