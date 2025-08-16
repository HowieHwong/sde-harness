"""Integration of LLMSR's existing evolutionary framework with SDE-Harness."""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import time
import copy
import dataclasses

# Add the original LLMSR path to import its components
llmsr_path = Path(__file__).parent.parent.parent.parent.parent / "llmsr"
print(llmsr_path)
sys.path.insert(0, str(llmsr_path))

try:
    from llmsr import buffer as llmsr_buffer
    from llmsr import code_manipulation
    from llmsr import config as llmsr_config
    from llmsr.buffer import ExperienceBuffer, Island, Cluster, Prompt as LLMSRPrompt
    from llmsr.code_manipulation import Program, Function
except ImportError as e:
    print(f"Warning: Could not import LLMSR components: {e}")
    print("Make sure the original LLMSR code is available at ../llmsr/")
    llmsr_buffer = None
    code_manipulation = None
    llmsr_config = None


class LLMSREvolutionaryIntegration:
    """Integrates LLMSR's existing evolutionary framework with SDE-Harness."""
    
    def __init__(self, 
                 num_islands: int = 4,
                 functions_per_prompt: int = 3,
                 cluster_sampling_temperature_init: float = 1.0,
                 cluster_sampling_temperature_period: int = 100,
                 reset_period: int = 100):
        """
        Initialize the LLMSR evolutionary integration.
        
        Args:
            num_islands: Number of islands in the evolutionary buffer
            functions_per_prompt: Number of functions to include in each prompt
            cluster_sampling_temperature_init: Initial temperature for Boltzmann sampling
            cluster_sampling_temperature_period: Period for temperature annealing
            reset_period: How often to reset weaker islands
        """
        if llmsr_buffer is None:
            raise ImportError("LLMSR components not available. Please ensure the original LLMSR code is accessible.")
        
        self.num_islands = num_islands
        self.functions_per_prompt = functions_per_prompt
        self.cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self.cluster_sampling_temperature_period = cluster_sampling_temperature_period
        self.reset_period = reset_period
        
        # LLMSR evolutionary buffer
        self.experience_buffer = None
        self.template = None
        self.function_to_evolve = "equation"
        
        # Tracking
        self.best_nmse_history = float('inf')
        self.current_iteration = 0
        self.last_reset_time = time.time()
    
    def initialize_buffer(self, var_names: List[str], var_descs: List[str], problem_name: str):
        """
        Initialize the LLMSR evolutionary buffer with the problem specification.
        
        Args:
            var_names: List of variable names
            var_descs: List of variable descriptions
            problem_name: Name of the problem
        """
        # Create the specification template similar to LLMSR searcher
        specification = self._create_specification(var_names, var_descs, problem_name)
        
        # Create LLMSR program template
        self.template = code_manipulation.text_to_program(specification)
        
        # Create LLMSR config
        config = llmsr_config.ExperienceBufferConfig(
            num_islands=self.num_islands,
            functions_per_prompt=self.functions_per_prompt,
            cluster_sampling_temperature_init=self.cluster_sampling_temperature_init,
            cluster_sampling_temperature_period=self.cluster_sampling_temperature_period,
            reset_period=self.reset_period
        )
        
        # Initialize the experience buffer
        self.experience_buffer = ExperienceBuffer(
            config=config,
            template=self.template,
            function_to_evolve=self.function_to_evolve
        )
        
        print(f"Initialized LLMSR evolutionary buffer with {self.num_islands} islands")
    
    def _create_specification(self, var_names: List[str], var_descs: List[str], problem_name: str) -> str:
        """Create the LLMSR specification template."""
        # Build input description
        if len(var_descs) > 2:
            input_desc = ", ".join(var_descs[1:-1]) + ", and " + var_descs[-1]
        else:
            input_desc = var_descs[-1]
        
        # Build function signature
        input_params = ", ".join([f"{name}: np.ndarray" for name in var_names[1:]])
        
        # Create specification similar to LLMSR searcher
        specification = '"""\n' + f"Find the mathematical function skeleton that represents {var_descs[0]}, given data on {input_desc}.\n" + '"""\n'
        
        # Add evaluation specification
        eval_spec = '''
import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs
    
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(*X, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss
'''
        specification += eval_spec
        
        # Create equation specification
        equation_specification = "@equation.evolve\n" + \
        "def equation(" + ", ".join([f"{name}: np.ndarray" for name in var_names[1:]]) + ", params: np.ndarray) -> np.ndarray:\n" + \
        f'    """ Mathematical function for {var_descs[0]}\n\n' + \
        '    Args:\n' + \
        "\n".join([f"        {name}: A numpy array representing observations of {desc}." for name, desc in zip(var_names[1:], var_descs[1:])]) + "\n" + \
        "        params: Array of numeric constants or parameters to be optimized\n\n" + \
        "    Return:\n" + \
        f"        A numpy array representing {var_descs[0]} as the result of applying the mathematical function to the inputs.\n" + \
        '    """\n' + \
        f"    {var_names[0]} = " + " + ".join([f"params[{i}] * {name}" for i, name in enumerate(var_names[1:])]) + f" + params[{len(var_names[1:])}]\n" + \
        f"    return {var_names[0]}"
        specification += equation_specification
        
        return specification
    
    def get_evolutionary_prompt(self) -> str:
        """
        Get a prompt from the LLMSR evolutionary buffer.
        
        Returns:
            The prompt string with evolutionary examples
        """
        if self.experience_buffer is None:
            raise ValueError("Evolutionary buffer not initialized. Call initialize_buffer() first.")
        
        # Get prompt from LLMSR buffer
        llmsr_prompt = self.experience_buffer.get_prompt()
        
        # Convert LLMSR prompt to string format suitable for SDE-Harness
        prompt_str = llmsr_prompt.code
        
        # Add evolutionary context
        prompt_str += "\nTry to improve upon the examples above."
        
        return prompt_str
    
    def add_program_to_buffer(self, 
                            code: str, 
                            nmse_score: float, 
                            iteration: int,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a program to the LLMSR evolutionary buffer.
        
        Args:
            code: The equation code
            nmse_score: NMSE score (lower is better)
            iteration: Current iteration number
            metadata: Additional metadata
            
        Returns:
            True if this program improved the global best score
        """
        if self.experience_buffer is None:
            return False
        
        if not np.isfinite(nmse_score):
            return False
        
        # Convert NMSE score to LLMSR format (higher is better)
        llmsr_score = -nmse_score  # Convert to negative so lower NMSE becomes higher score
        
        # Create LLMSR function object
        try:
            # Extract the equation function from the code
            equation_code = self._extract_equation_function(code)
            if equation_code is None:
                return False
            
            # Create LLMSR function
            llmsr_function = code_manipulation.text_to_function(equation_code)
            
            # Create scores per test (LLMSR format)
            scores_per_test = {"nmse": llmsr_score}
            
            # Register in buffer
            self.experience_buffer.register_program(
                program=llmsr_function,
                island_id=None,  # Let LLMSR choose island
                scores_per_test=scores_per_test,
                profiler=None,
                global_sample_nums=iteration,
                sample_time=0,
                evaluate_time=0
            )
            
            # Check if this improves global best
            improved = nmse_score < self.best_nmse_history
            
            if improved:
                self.best_nmse_history = nmse_score
                print(f"New best NMSE achieved: {nmse_score:.6f}")
            
            self.current_iteration = iteration
            
            return improved
            
        except Exception as e:
            print(f"Error adding program to buffer: {e}")
            return False
    
    def _extract_equation_function(self, code: str) -> Optional[str]:
        """Extract the equation function from the generated code."""
        # Look for the equation function definition
        lines = code.split('\n')
        in_function = False
        function_lines = []
        
        for line in lines:
            if 'def equation(' in line:
                in_function = True
                function_lines.append(line)
            elif in_function:
                if line.strip() == '' or line.startswith('def '):
                    break
                function_lines.append(line)
        
        if function_lines:
            return '\n'.join(function_lines)
        return None
    
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get statistics about the LLMSR evolutionary buffer."""
        if self.experience_buffer is None:
            return {"buffer_initialized": False}
        
        total_programs = sum(len(island._clusters) for island in self.experience_buffer._islands)
        non_empty_islands = sum(1 for island in self.experience_buffer._islands if island._clusters)
        
        return {
            "buffer_initialized": True,
            "total_programs": total_programs,
            "non_empty_islands": non_empty_islands,
            "best_nmse_history": self.best_nmse_history,
            "current_iteration": self.current_iteration,
            "num_islands": self.num_islands,
            "functions_per_prompt": self.functions_per_prompt,
            "island_stats": [
                {
                    "island_id": i,
                    "num_clusters": len(island._clusters),
                    "best_score": -self.experience_buffer._best_score_per_island[i] if self.experience_buffer._best_score_per_island[i] > -float('inf') else float('inf')
                }
                for i, island in enumerate(self.experience_buffer._islands)
            ]
        }
    
    def clear_buffer(self):
        """Clear the LLMSR evolutionary buffer."""
        if self.experience_buffer:
            # Reinitialize the buffer
            config = llmsr_config.ExperienceBufferConfig(
                num_islands=self.num_islands,
                functions_per_prompt=self.functions_per_prompt,
                cluster_sampling_temperature_init=self.cluster_sampling_temperature_init,
                cluster_sampling_temperature_period=self.cluster_sampling_temperature_period,
                reset_period=self.reset_period
            )
            
            self.experience_buffer = ExperienceBuffer(
                config=config,
                template=self.template,
                function_to_evolve=self.function_to_evolve
            )
        
        self.best_nmse_history = float('inf')
        self.current_iteration = 0
        self.last_reset_time = time.time()
    
    def should_include_example(self, nmse_score: float) -> bool:
        """
        Determine if an example should be included based on NMSE score.
        
        Args:
            nmse_score: The NMSE score to check
            
        Returns:
            True if the example should be included
        """
        # Only include examples that improve upon the best score
        return nmse_score < self.best_nmse_history


