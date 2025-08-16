"""LLMSR Evolutionary Prompt Templates - Integration with original LLMSR framework."""

from typing import Dict, Any, List, Optional
from sde_harness.core import Prompt

from core.llmsr_evolutionary_integration import LLMSREvolutionaryIntegration


class LLMSREvolutionaryPromptTemplates:
    """Collection of LLMSR evolutionary prompt templates using original LLMSR framework."""
    
    def __init__(self, 
                 num_islands: int = 4,
                 functions_per_prompt: int = 3,
                 cluster_sampling_temperature_init: float = 1.0,
                 cluster_sampling_temperature_period: int = 100,
                 reset_period: int = 100):
        """
        Initialize LLMSR evolutionary prompt templates.
        
        Args:
            num_islands: Number of islands in the evolutionary buffer
            functions_per_prompt: Number of functions to include in each prompt
            cluster_sampling_temperature_init: Initial temperature for Boltzmann sampling
            cluster_sampling_temperature_period: Period for temperature annealing
            reset_period: How often to reset weaker islands
        """
        self.llmsr_integration = LLMSREvolutionaryIntegration(
            num_islands=num_islands,
            functions_per_prompt=functions_per_prompt,
            cluster_sampling_temperature_init=cluster_sampling_temperature_init,
            cluster_sampling_temperature_period=cluster_sampling_temperature_period,
            reset_period=reset_period
        )
    
    @staticmethod
    def create_base_prompt(var_names: List[str], var_descs: List[str], 
                          problem_name: str, expression: str = "", iteration: int = 1) -> Prompt:
        """
        Create base prompt for equation discovery.
        
        Args:
            var_names: List of variable names
            var_descs: List of variable descriptions
            problem_name: Name of the problem
            expression: Ground truth expression (optional)
            
        Returns:
            Prompt instance
        """
        # Build input description
        if len(var_descs) > 2:
            input_desc = ", ".join(var_descs[1:-1]) + ", and " + var_descs[-1]
        else:
            input_desc = var_descs[-1]
        
        # Build function signature
        input_params = ", ".join([f"{name}: np.ndarray" for name in var_names[1:]])
        
        if iteration == 1:
            # Create the prompt template
            template = f"""
                You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                Your task is to find the mathematical function skeleton that represents {var_descs[0]}, given data on {input_desc}. \
                Consider the physical meaning and relationships of inputs in finding the mathematical relations between variables. \

                Here's the function signature you should use:

                ```python
                import numpy as np

                def equation({input_params}, params: np.ndarray) -> np.ndarray:
                    \"\"\" Mathematical function for {var_descs[0]}
                    
                    Args:
                {chr(10).join([f"        {name}: A numpy array representing observations of {desc}." for name, desc in zip(var_names[1:], var_descs[1:])])}
                        params: Array of numeric constants or parameters to be optimized
                    
                    Returns:
                        A numpy array representing {var_descs[0]} as the result of applying the mathematical function to the inputs.
                    \"\"\"
                    # TODO: Implement the mathematical relationship here""" + \
                    "# Example: {var_names[0]} = " + " + ".join([f"params[{i}] * {name}" for i, name in enumerate(var_names[1:])]) + f" + params[{len(var_names[1:])}]\n" + \
                    """# Replace with your discovered equation
                    
                    return {var_names[0]}
                ```

                Explain your reasoning briefly before giving the complete Python function implementation. Let's think step by step.
                """
        else:
            template = f"""
                You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                Your task is to find the mathematical function skeleton that represents {var_descs[0]}, given data on {input_desc}.
                Consider the physical meaning and relationships of inputs in finding the mathematical relations between variables. \

                {{llmsr_evolutionary_section}}
                """
        
        return Prompt(
            custom_template=template,
            default_vars={
                "var_names": var_names,
                "var_descs": var_descs,
                "problem_name": problem_name,
                "expression": expression,
                "llmsr_evolutionary_section": ""
            }
        )
    
    def create_llmsr_evolutionary_prompt(self, 
                                       var_names: List[str], 
                                       var_descs: List[str], 
                                       problem_name: str, 
                                       current_iteration: int) -> Prompt:
        """
        Create LLMSR evolutionary prompt with examples from the original LLMSR framework.
        
        Args:
            var_names: List of variable names
            var_descs: List of variable descriptions
            problem_name: Name of the problem
            current_iteration: Current iteration number
            
        Returns:
            Prompt instance with LLMSR evolutionary examples
        """
        # Initialize buffer if not already done
        if self.llmsr_integration.experience_buffer is None:
            self.llmsr_integration.initialize_buffer(var_names, var_descs, problem_name)
        
        base_prompt = self.create_base_prompt(
            var_names, var_descs, problem_name, iteration=current_iteration
        )
        
        # Get evolutionary prompt from LLMSR buffer
        try:
            evolutionary_section = self.llmsr_integration.get_evolutionary_prompt()
            
            # If this is not the first iteration and we have examples, use them
            if current_iteration > 1 and "Current best NMSE" in evolutionary_section:
                base_prompt.add_vars(llmsr_evolutionary_section=evolutionary_section)
            else:
                # For first iteration, just use base prompt
                base_prompt.add_vars(llmsr_evolutionary_section="")
                
        except Exception as e:
            print(f"Warning: Could not get evolutionary prompt: {e}")
            base_prompt.add_vars(llmsr_evolutionary_section="")
        
        return base_prompt
    
    def create_dynamic_llmsr_evolutionary_prompt_function(self, 
                                                        var_names: List[str], 
                                                        var_descs: List[str], 
                                                        problem_name: str) -> callable:
        """
        Create a dynamic LLMSR evolutionary prompt function.
        
        Args:
            var_names: List of variable names
            var_descs: List of variable descriptions
            problem_name: Name of the problem
            
        Returns:
            Function that takes iteration, history and returns a Prompt
        """
        def dynamic_llmsr_evolutionary_prompt_fn(iteration: int, history: Dict[str, List[Any]]) -> Prompt:
            # Add previous iteration's result to buffer if it exists
            if iteration > 1 and history.get("outputs") and history.get("scores"):
                prev_output = history["outputs"][-1]  # Previous iteration's output
                prev_scores = history["scores"][-1]   # Previous iteration's scores
                
                if isinstance(prev_scores, dict) and "nmse" in prev_scores:
                    prev_nmse = prev_scores["nmse"]

                    from core.generation import LLMSRGeneration
                    generator = LLMSRGeneration()
                    prev_code = generator.parse_equation_code(prev_output)
                    if isinstance(prev_code, list):
                        prev_code = prev_code[0]
                        prev_code = "```python\n" + prev_code + "\n```"
                    else:
                        prev_code = prev_output
                    
                    # Add to LLMSR evolutionary buffer
                    self.llmsr_integration.add_program_to_buffer(
                        code=prev_code,
                        nmse_score=prev_nmse,
                        iteration=iteration-1,
                        metadata={
                            "problem_name": problem_name,
                            "var_names": var_names,
                            "var_descs": var_descs
                        }
                    )
            
            # Create LLMSR evolutionary prompt
            return self.create_llmsr_evolutionary_prompt(
                var_names=var_names,
                var_descs=var_descs,
                problem_name=problem_name,
                current_iteration=iteration
            )
        
        return dynamic_llmsr_evolutionary_prompt_fn
    
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
        return self.llmsr_integration.add_program_to_buffer(
            code=code,
            nmse_score=nmse_score,
            iteration=iteration,
            metadata=metadata
        )
    
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get statistics about the LLMSR evolutionary buffer."""
        return self.llmsr_integration.get_buffer_statistics()
    
    def clear_buffer(self):
        """Clear the LLMSR evolutionary buffer."""
        self.llmsr_integration.clear_buffer()
    
    def should_include_example(self, nmse_score: float) -> bool:
        """
        Determine if an example should be included based on NMSE score.
        
        Args:
            nmse_score: The NMSE score to check
            
        Returns:
            True if the example should be included
        """
        return self.llmsr_integration.should_include_example(nmse_score)


