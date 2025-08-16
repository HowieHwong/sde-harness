"""Evolutionary workflow for LLMSR with selective example inclusion."""

import os
import asyncio
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sde_harness.core import Workflow
from core import LLMSRGeneration
from oracles import EquationOracle
from data import LLMSRDatasetLoader, EquationData
from modes.evolutionary_prompt_templates import EvolutionaryPromptTemplates


class LLMSREvolutionaryWorkflow(Workflow):
    """Evolutionary workflow for equation discovery using LLMs with selective example inclusion."""
    
    def __init__(self, 
                 model_name: str = "openai/gpt-4o-2024-08-06",
                 max_iterations: int = 5,
                 max_params: int = 10,
                 optimization_method: str = 'BFGS',
                 num_islands: int = 4,
                 max_programs_per_island: int = 50,
                 reset_period: int = 100,
                 reset_fraction: float = 0.5,
                 num_examples: int = 3,
                 temperature: float = 1.0,
                 sampling_strategy: str = "boltzmann"):
        """
        Initialize the evolutionary LLMSR workflow.
        
        Args:
            model_name: Name of the LLM model to use
            max_iterations: Maximum number of optimization iterations
            max_params: Maximum number of parameters to optimize
            optimization_method: Optimization method for parameter fitting
            num_islands: Number of islands in the evolutionary buffer
            max_programs_per_island: Maximum programs per island
            reset_period: How often to reset weaker islands
            reset_fraction: Fraction of islands to reset
            num_examples: Number of examples to include in prompts
            temperature: Temperature for Boltzmann sampling
            sampling_strategy: Sampling strategy ("boltzmann", "best", "random")
        """
        # Initialize components
        generator = LLMSRGeneration(model_name=model_name)
        oracle = EquationOracle(max_params=max_params, optimization_method=optimization_method)
        
        super().__init__(
            generator=generator,
            oracle=oracle,
            max_iterations=max_iterations,
            enable_history_in_prompts=False,  # We handle history manually in the prompt function
            enable_multi_round_metrics=True
        )
        
        # Initialize evolutionary components
        self.evolutionary_templates = EvolutionaryPromptTemplates(
            num_islands=num_islands,
            max_programs_per_island=max_programs_per_island,
            reset_period=reset_period,
            reset_fraction=reset_fraction
        )
        
        # Evolutionary parameters
        self.num_examples = num_examples
        self.temperature = temperature
        self.sampling_strategy = sampling_strategy
        
        # Tracking
        self.dataset_loader = None
        self.results = {}
        self.current_problem_name = None
    
    def setup_dataset(self, dataset_name: str = "lsrtransform"):
        """
        Setup the dataset for equation discovery.
        
        Args:
            dataset_name: Name of the dataset to load
        """
        self.dataset_loader = LLMSRDatasetLoader(dataset_name)
        self.dataset_loader.setup()
    
    def discover_equation_evolutionary(self, 
                                     problem: Union[EquationData, str],
                                     output_dir: str = "outputs") -> Dict[str, Any]:
        """
        Discover equations using evolutionary approach with selective example inclusion.
        
        Args:
            problem: Either an EquationData object or problem name string
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing discovery results
        """
        # Get problem data
        if isinstance(problem, str):
            if self.dataset_loader is None:
                raise ValueError("Dataset not loaded. Call setup_dataset() first.")
            problem = self.dataset_loader.get_problem_by_name(problem)
            if problem is None:
                raise ValueError(f"Problem '{problem}' not found in dataset.")
        
        # Get formatted data
        problem_data = self.dataset_loader.get_problem_data(problem)
        self.current_problem_name = problem_data['problem_name']
        
        # Clear evolutionary buffer for new problem
        self.evolutionary_templates.clear_buffer()
        
        # Create dynamic evolutionary prompt function
        dynamic_prompt_fn = self.evolutionary_templates.create_dynamic_evolutionary_prompt_function(
            problem_data['var_names'],
            problem_data['var_descs'],
            problem_data['problem_name'],
            num_examples=self.num_examples,
            temperature=self.temperature,
            sampling_strategy=self.sampling_strategy
        )
        
        # Create output directory
        output_path = Path(output_dir) / problem_data['problem_name']
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting evolutionary equation discovery for problem: {problem_data['problem_name']}")
        print(f"Variables: {problem_data['var_names']}")
        print(f"Descriptions: {problem_data['var_descs']}")
        print(f"Evolutionary parameters: {self.num_examples} examples, temperature={self.temperature}, strategy={self.sampling_strategy}")
        
        # Run the workflow synchronously
        result = self.run_sync(
            prompt=dynamic_prompt_fn,
            reference=problem_data,
            gen_args={
                "max_tokens": 2000,
                "temperature": 1.0
            },
            metrics=["nmse", "rmse", "mae", "improvement_rate", "convergence_score"],
            history_context={
                "task_description": f"Evolutionary equation discovery for {problem_data['problem_name']}",
                "var_names": problem_data['var_names'],
                "var_descs": problem_data['var_descs']
            }
        )
        
        # Add evolutionary statistics to results
        result["evolutionary_statistics"] = self.evolutionary_templates.get_buffer_statistics()
        
        # Save results
        self.results[problem_data['problem_name']] = result
        
        # Save to file
        result_file = output_path / "evolutionary_discovery_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Evolutionary results saved to: {result_file}")
        
        return result
    
    def discover_all_equations_evolutionary(self, 
                                          output_dir: str = "outputs",
                                          problem_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Discover equations for all problems using evolutionary approach.
        
        Args:
            output_dir: Directory to save results
            problem_filter: Optional list of problem names to process
            
        Returns:
            Dictionary containing results for all problems
        """
        if self.dataset_loader is None:
            raise ValueError("Dataset not loaded. Call setup_dataset() first.")
        
        problems = self.dataset_loader.problems
        
        if problem_filter:
            problems = [p for p in problems if p.name in problem_filter]
        
        print(f"Starting evolutionary equation discovery for {len(problems)} problems")
        
        all_results = {}
        
        for i, problem in enumerate(problems):
            print(f"\nProcessing problem {i+1}/{len(problems)}: {problem.name}")
            
            try:
                result = self.discover_equation_evolutionary(problem, output_dir)
                all_results[problem.name] = result
                
                # Print summary
                best_score = min([s.get("nmse", float('inf')) for s in result.get("scores", [])])
                print(f"Best NMSE: {best_score}")
                
                # Print evolutionary statistics
                evo_stats = result.get("evolutionary_statistics", {})
                print(f"Evolutionary stats: {evo_stats.get('total_programs', 0)} programs, "
                      f"{evo_stats.get('programs_improved', 0)} improvements")
                
            except Exception as e:
                print(f"Error processing problem {problem.name}: {e}")
                all_results[problem.name] = {"error": str(e)}
        
        # Save overall results
        overall_file = Path(output_dir) / "all_evolutionary_results.json"
        with open(overall_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\nAll evolutionary results saved to: {overall_file}")
        
        return all_results
    
    def get_best_equation(self, problem_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the best equation found for a problem using evolutionary approach.
        
        Args:
            problem_name: Name of the problem
            
        Returns:
            Dictionary containing best equation info or None
        """
        if problem_name not in self.results:
            return None
        
        result = self.results[problem_name]
        scores = result.get("scores", [])
        
        if not scores:
            return None
        
        # Find best score
        best_idx = 0
        best_nmse = float('inf')
        
        for i, score in enumerate(scores):
            nmse = score.get("nmse", float('inf'))
            if nmse < best_nmse:
                best_nmse = nmse
                best_idx = i
        
        return {
            "equation_code": result["outputs"][best_idx],
            "nmse_score": best_nmse,
            "iteration": best_idx + 1,
            "all_scores": scores[best_idx],
            "evolutionary_statistics": result.get("evolutionary_statistics", {})
        }
    
    def print_summary(self):
        """Print a summary of all evolutionary results."""
        if not self.results:
            print("No evolutionary results available.")
            return
        
        print("\n" + "="*80)
        print("EVOLUTIONARY EQUATION DISCOVERY SUMMARY")
        print("="*80)
        
        for problem_name, result in self.results.items():
            best_equation = self.get_best_equation(problem_name)
            
            if best_equation:
                print(f"\nProblem: {problem_name}")
                print(f"Best NMSE: {best_equation['nmse_score']:.6f}")
                print(f"Found in iteration: {best_equation['iteration']}")
                
                # Extract equation body
                from core.generation import LLMSRGeneration
                generator = LLMSRGeneration()
                equation_body = generator.extract_equation_body(best_equation['equation_code'])
                print(f"Equation: {equation_body}")
                
                # Print evolutionary statistics
                evo_stats = best_equation.get("evolutionary_statistics", {})
                print(f"Evolutionary stats: {evo_stats.get('total_programs', 0)} programs, "
                      f"{evo_stats.get('programs_improved', 0)} improvements, "
                      f"improvement rate: {evo_stats.get('improvement_rate', 0):.3f}")
            else:
                print(f"\nProblem: {problem_name}")
                print("No valid equation found")
        
        print("\n" + "="*80)
    
    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare evolutionary results with baseline results.
        
        Args:
            baseline_results: Results from baseline approach
            
        Returns:
            Comparison results
        """
        comparison = {
            "problems_compared": 0,
            "evolutionary_better": 0,
            "baseline_better": 0,
            "same_performance": 0,
            "detailed_comparison": {}
        }
        
        for problem_name in self.results:
            if problem_name in baseline_results:
                evo_best = self.get_best_equation_evolutionary(problem_name)
                baseline_best = baseline_results[problem_name].get("best_equation", {})
                
                if evo_best and baseline_best:
                    evo_nmse = evo_best["nmse_score"]
                    baseline_nmse = baseline_best.get("nmse_score", float('inf'))
                    
                    comparison["problems_compared"] += 1
                    
                    if evo_nmse < baseline_nmse:
                        comparison["evolutionary_better"] += 1
                    elif baseline_nmse < evo_nmse:
                        comparison["baseline_better"] += 1
                    else:
                        comparison["same_performance"] += 1
                    
                    comparison["detailed_comparison"][problem_name] = {
                        "evolutionary_nmse": evo_nmse,
                        "baseline_nmse": baseline_nmse,
                        "improvement": baseline_nmse - evo_nmse,
                        "evolutionary_better": evo_nmse < baseline_nmse
                    }
        
        return comparison
    
    def get_evolutionary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolutionary statistics."""
        if not self.results:
            return {"message": "No evolutionary results available"}
        
        all_stats = []
        total_programs = 0
        total_improvements = 0
        
        for problem_name, result in self.results.items():
            evo_stats = result.get("evolutionary_statistics", {})
            all_stats.append({
                "problem_name": problem_name,
                **evo_stats
            })
            total_programs += evo_stats.get("total_programs", 0)
            total_improvements += evo_stats.get("programs_improved", 0)
        
        return {
            "total_problems": len(self.results),
            "total_programs_generated": total_programs,
            "total_improvements": total_improvements,
            "overall_improvement_rate": total_improvements / max(1, total_programs),
            "per_problem_statistics": all_stats,
            "average_programs_per_problem": total_programs / max(1, len(self.results)),
            "average_improvements_per_problem": total_improvements / max(1, len(self.results))
        }
