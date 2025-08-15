"""Prompt templates for equation discovery tasks."""

from typing import Dict, Any, List, Optional
from sde_harness.core import Prompt


class EquationPromptTemplates:
    """Collection of prompt templates for equation discovery."""
    
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
        
        if iteration == 0:
            # Create the prompt template
            template = f"""
                You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                Your task is to find the mathematical function skeleton that represents {var_descs[0]}, given data on {input_desc}.

                Here's the function signature you should use:

                ```python
                import numpy as np

                MAX_NPARAMS = 10
                params = [1.0]*MAX_NPARAMS

                def equation({input_params}, params: np.ndarray) -> np.ndarray:
                    \"\"\" Mathematical function for {var_descs[0]}
                    
                    Args:
                {chr(10).join([f"        {name}: A numpy array representing observations of {desc}." for name, desc in zip(var_names[1:], var_descs[1:])])}
                        params: Array of numeric constants or parameters to be optimized
                    
                    Returns:
                        A numpy array representing {var_descs[0]} as the result of applying the mathematical function to the inputs.
                    \"\"\"
                    # TODO: Implement the mathematical relationship here
                    # Example: {var_names[0]} = params[0] * {var_names[1]} + params[1]
                    # Replace with your discovered equation
                    
                    return {var_names[0]}
                ```

                Please provide only the Python function implementation. Consider the physical meaning and relationships of inputs in finding the mathematical relations between variables.
                """
        else:
            template = f"""
                You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                Your task is to find the mathematical function skeleton that represents {var_descs[0]}, given data on {input_desc}.

                {{history_section}}
                """
        
        return Prompt(
            custom_template=template,
            default_vars={
                "var_names": var_names,
                "var_descs": var_descs,
                "problem_name": problem_name,
                "expression": expression,
                "history_section": ""
            }
        )
    
    @staticmethod
    def create_iterative_prompt(var_names: List[str], var_descs: List[str], 
                              problem_name: str, history: Dict[str, List[Any]], 
                              current_iteration: int) -> Prompt:
        """
        Create iterative prompt with history.
        
        Args:
            var_names: List of variable names
            var_descs: List of variable descriptions
            problem_name: Name of the problem
            history: Dictionary containing previous attempts
            current_iteration: Current iteration number
            
        Returns:
            Prompt instance with history
        """
        base_prompt = EquationPromptTemplates.create_base_prompt(
            var_names, var_descs, problem_name
        )
        

        # Add history information
        if history.get("outputs") and len(history["outputs"]) > 0:
            history_section = "\n\nPrevious attempts and their performance:\n"
            
            for i, (output, scores) in enumerate(zip(history["outputs"], history["scores"])):
                iteration_num = i + 1
                nmse_score = scores.get("nmse", "N/A") if isinstance(scores, dict) else scores
                
                # Extract equation body from previous attempt
                from core.generation import LLMSRGeneration
                generator = LLMSRGeneration()
                equation_body = generator.extract_equation_body(output)
                
                history_section += f"\nIteration {iteration_num}:\n"
                history_section += f"Equation: {equation_body}\n"
                history_section += f"NMSE Score: {nmse_score}\n"
            
            # Add improvement instructions
            if current_iteration > 1:
                history_section += f"\nBased on the previous attempts, try to improve the equation. Please provide the Python function of improved equation."
            
            base_prompt.add_vars(history_section=history_section)
        
        return base_prompt
    
    @staticmethod
    def create_dynamic_prompt_function(var_names: List[str], var_descs: List[str], 
                                     problem_name: str) -> callable:
        """
        Create a dynamic prompt function for iterative optimization.
        
        Args:
            var_names: List of variable names
            var_descs: List of variable descriptions
            problem_name: Name of the problem
            
        Returns:
            Function that takes iteration and history and returns a Prompt
        """
        def dynamic_prompt_fn(iteration: int, history: Dict[str, List[Any]]) -> Prompt:
            # Create base prompt
            prompt = EquationPromptTemplates.create_base_prompt(
                var_names, var_descs, problem_name, iteration=len(history["outputs"])
            )
            
            # Add history information if available
            if history.get("outputs") and len(history["outputs"]) > 0:
                history_section = "\n\nPrevious attempts and their performance:\n"
                
                for i, (output, scores) in enumerate(zip(history["outputs"], history["scores"])):
                    iteration_num = i + 1
                    nmse_score = scores.get("nmse", "N/A") if isinstance(scores, dict) else scores
                    
                    # Extract equation body from previous attempt
                    from core.generation import LLMSRGeneration
                    generator = LLMSRGeneration()
                    equations = generator.parse_equation_code(output)
                    if len(equations) == 0:
                        return float('inf')
                    equation_code = equations[0]
                    
                    history_section += f"\nIteration {iteration_num}:\n"
                    history_section += f"Equation: {equation_code}\n"
                    history_section += f"NMSE Score: {nmse_score}\n"
                
                # Add improvement instructions for iterations after the first
                if iteration > 1:
                    history_section += f"\nBased on the previous attempts, try to improve the equation. Please provide the Python function of improved equation."

                prompt.add_vars(history_section=history_section)
            
            return prompt
        
        return dynamic_prompt_fn
