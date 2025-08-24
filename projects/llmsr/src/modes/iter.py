"""Synchronous version of LLMSR workflow for easier debugging."""

import os
import asyncio
from typing import Dict, Any, List, Optional, Union
import json
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sde_harness.core import Workflow
from ..core.generation import LLMSRGeneration
from ..core.oracle import EquationOracle
from ..core.prompt import PromptTemplates

project_root2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root2 not in sys.path:
    sys.path.insert(0, project_root2)
from data.dataset import LLMSRDatasetLoader, EquationData





class LLMSRWorkflow(Workflow):
    """Synchronous workflow for equation discovery using LLMs - easier to debug."""
    
    def __init__(self, 
                 model_name: str = "openai/gpt-4o-2024-08-06",
                 max_iterations: int = 5,
                 max_params: int = 10,
                 optimization_method: str = 'BFGS'):
        """
        Initialize the LLMSR workflow.
        
        Args:
            model_name: Name of the LLM model to use
            max_iterations: Maximum number of optimization iterations
            max_params: Maximum number of parameters to optimize
            optimization_method: Optimization method for parameter fitting
        """
        # Initialize components
        generator = LLMSRGeneration(model_name=model_name)
        oracle = EquationOracle(max_params=max_params, optimization_method=optimization_method)
        
        super().__init__(
            generator=generator,
            oracle=oracle,
            max_iterations=max_iterations,
            enable_history_in_prompts=False, # We handle history manually in the prompt function
            enable_multi_round_metrics=True
        )
        
        self.dataset_loader = None
        self.results = {}
    
    def setup_dataset(self, dataset_name: str = "lsrtransform"):
        """
        Setup the dataset for equation discovery.
        
        Args:
            dataset_name: Name of the dataset to load
        """
        self.dataset_loader = LLMSRDatasetLoader(dataset_name)
        self.dataset_loader.setup()
    
    def discover_equation_sync(self, 
                             problem: Union[EquationData, str],
                             output_dir: str = "outputs") -> Dict[str, Any]:
        """
        Synchronous version of equation discovery for easier debugging.
        
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
        
        # Create dynamic prompt function
        dynamic_prompt_fn = PromptTemplates.create_dynamic_prompt_function(
            problem_data['var_names'],
            problem_data['var_descs'],
            problem_data['problem_name']
        )
        
        # Create output directory
        output_path = Path(output_dir) / problem_data['problem_name']
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting equation discovery for problem: {problem_data['problem_name']}")
        print(f"Variables: {problem_data['var_names']}")
        print(f"Descriptions: {problem_data['var_descs']}")
        
        
        # Run the workflow synchronously
        result = self.run_sync(
            prompt=dynamic_prompt_fn,
            reference=problem_data,
            gen_args={
                "max_tokens": 2000,
                "temperature": 1.0
            },
            metrics=["nmse", "rmse", "mae", "improvement_rate", "convergence_score"],
            # metrics=["nmse"],
            history_context={
                "task_description": f"Equation discovery for {problem_data['problem_name']}",
                "var_names": problem_data['var_names'],
                "var_descs": problem_data['var_descs']
            }
        )
        
        # Save results
        self.results[problem_data['problem_name']] = result
        
        # Save to file
        result_file = output_path / "discovery_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Results saved to: {result_file}")
        
        return result
    
    def discover_all_equations_sync(self, 
                                  output_dir: str = "outputs",
                                  problem_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Synchronous version of discovering equations for all problems.
        
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
        
        print(f"Starting equation discovery for {len(problems)} problems")
        
        all_results = {}
        
        for i, problem in enumerate(problems):
            print(f"\nProcessing problem {i+1}/{len(problems)}: {problem.name}")
            
            try:
                result = self.discover_equation_sync(problem, output_dir)
                all_results[problem.name] = result
                
                # Print summary
                best_score = min([s.get("nmse", float('inf')) for s in result.get("scores", [])])
                print(f"Best NMSE: {best_score}")
                
            except Exception as e:
                print(f"Error processing problem {problem.name}: {e}")
                all_results[problem.name] = {"error": str(e)}
        
        # Save overall results
        overall_file = Path(output_dir) / "all_results.json"
        with open(overall_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\nAll results saved to: {overall_file}")
        
        return all_results
    
    def get_best_equation(self, problem_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the best equation found for a problem.
        
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
            "all_scores": scores[best_idx]
        }
    
    def print_summary(self):
        """Print a summary of all results."""
        if not self.results:
            print("No results available.")
            return
        
        print("\n" + "="*60)
        print("EQUATION DISCOVERY SUMMARY")
        print("="*60)
        
        for problem_name, result in self.results.items():
            best_equation = self.get_best_equation(problem_name)
            
            if best_equation:
                print(f"\nProblem: {problem_name}")
                print(f"Best NMSE: {best_equation['nmse_score']:.6f}")
                print(f"Found in iteration: {best_equation['iteration']}")
                
                # Extract equation body
                from ..core.generation import LLMSRGeneration
                generator = LLMSRGeneration()
                equation_body = generator.extract_equation_body(best_equation['equation_code'])
                print(f"Equation: {equation_body}")
            else:
                print(f"\nProblem: {problem_name}")
                print("No valid equation found")
        
        print("\n" + "="*60)


