"""Prompt templates for equation discovery tasks."""

from typing import Dict, Any, List, Optional
from sde_harness.core import Prompt

from .buffer import EvolutionaryBuffer, EquationProgram


class PromptTemplates:
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
                        A numpy array representing {var_names[0]} as the result of applying the mathematical function to the inputs.
                    \"\"\"
                    # TODO: Implement the mathematical relationship here""" + \
                    "# Example: {var_names[0]} = " + " + ".join([f"params[{i}] * {name}" for i, name in enumerate(var_names[1:])]) + f" + params[{len(var_names[1:])}]\n" + \
                    """# Replace with your discovered equation
                    
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
        base_prompt = PromptTemplates.create_base_prompt(
            var_names, var_descs, problem_name
        )
        

        # Add history information
        if history.get("outputs") and len(history["outputs"]) > 0:
            history_section = "\n\nPrevious attempts and their performance:\n"
            
            for i, (output, scores) in enumerate(zip(history["outputs"], history["scores"])):
                iteration_num = i + 1
                nmse_score = scores.get("nmse", "N/A") if isinstance(scores, dict) else scores
                
                # Extract equation body from previous attempt
                from .generation import LLMSRGeneration
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
            prompt = PromptTemplates.create_base_prompt(
                var_names, var_descs, problem_name, iteration=len(history["outputs"])
            )
            
            # Add history information if available
            if history.get("outputs") and len(history["outputs"]) > 0:
                history_section = "\n\nPrevious attempts and their performance:\n"
                
                for i, (output, scores) in enumerate(zip(history["outputs"], history["scores"])):
                    iteration_num = i + 1
                    nmse_score = scores.get("nmse", "N/A") if isinstance(scores, dict) else scores
                    
                    # Extract equation body from previous attempt
                    from .generation import LLMSRGeneration
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



class EvolutionaryPromptTemplates:
    """Collection of evolutionary prompt templates for equation discovery."""
    
    def __init__(self, 
                 num_islands: int = 4,
                 max_programs_per_island: int = 50,
                 reset_period: int = 100,
                 reset_fraction: float = 0.5):
        """
        Initialize evolutionary prompt templates.
        
        Args:
            num_islands: Number of islands in the evolutionary buffer
            max_programs_per_island: Maximum programs per island
            reset_period: How often to reset weaker islands
            reset_fraction: Fraction of islands to reset
        """
        self.evolutionary_buffer = EvolutionaryBuffer(
            num_islands=num_islands,
            max_programs_per_island=max_programs_per_island,
            reset_period=reset_period,
            reset_fraction=reset_fraction
        )
        self.best_nmse_history = float('inf')
    
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
                        A numpy array representing {var_names[0]} as the result of applying the mathematical function to the inputs.
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

                {{evolutionary_examples_section}}
                """
        
        return Prompt(
            custom_template=template,
            default_vars={
                "var_names": var_names,
                "var_descs": var_descs,
                "problem_name": problem_name,
                "expression": expression,
                "evolutionary_examples_section": ""
            }
        )
    
    def create_evolutionary_prompt(self, 
                                 var_names: List[str], 
                                 var_descs: List[str], 
                                 problem_name: str, 
                                 current_iteration: int,
                                 current_nmse: float,
                                 num_examples: int = 3,
                                 temperature: float = 1.0,
                                 sampling_strategy: str = "boltzmann") -> Prompt:
        """
        Create evolutionary prompt with selective examples.
        
        Args:
            var_names: List of variable names
            var_descs: List of variable descriptions
            problem_name: Name of the problem
            current_iteration: Current iteration number
            current_nmse: Current NMSE score
            num_examples: Number of examples to include
            temperature: Temperature for Boltzmann sampling
            sampling_strategy: Sampling strategy ("boltzmann", "best", "random")
            
        Returns:
            Prompt instance with evolutionary examples
        """
        base_prompt = self.create_base_prompt(
            var_names, var_descs, problem_name, iteration=current_iteration
        )
        
        # Check if current NMSE improves the best score
        improved = current_nmse < self.best_nmse_history
        
        if improved:
            # Update best NMSE history
            self.best_nmse_history = current_nmse
            print(f"New best NMSE achieved: {current_nmse:.6f}")
        
        # Sample examples from evolutionary buffer
        examples = self.evolutionary_buffer.sample_examples(
            num_examples=num_examples,
            temperature=temperature,
            strategy=sampling_strategy
        )
        
        # Create evolutionary examples section
        if examples:
            examples_section = "\n\nHere are some successful examples of equation Python function implementations to learn from:\n"
            
            for i, example in enumerate(examples):
                # Extract equation body from example
                from .generation import LLMSRGeneration
                generator = LLMSRGeneration()
                
                examples_section += f"\nExample {i+1} (NMSE: {example.nmse_score:.6f}):\n"
                examples_section += f"{example.code}\n"
            
            examples_section += f"\nBased on the previous attempts, try to improve the equation with similar function signature. \
                                    Explain your reasoning briefly before giving the complete Python function implementation of improved equation. Let's think step by step."

            input_params = ", ".join([f"{name}: np.ndarray" for name in var_names[1:]])
            examples_section += f"""Here's the function signature you should use:

                                        ```python
                                        import numpy as np

                                        def equation({input_params}, params: np.ndarray) -> np.ndarray:
                                            \"\"\" Mathematical function for {var_descs[0]}
                                            
                                            Args:
                                        {chr(10).join([f"        {name}: A numpy array representing observations of {desc}." for name, desc in zip(var_names[1:], var_descs[1:])])}
                                                params: Array of numeric constants or parameters to be optimized
                                            
                                            Returns:
                                                A numpy array representing {var_names[0]} as the result of applying the mathematical function to the inputs.
                                            \"\"\"
                                            # TODO: Implement your mathematical relationship here 
                                            ...
                                            return {var_names[0]}
                                        ```
                                        """
            
            base_prompt.add_vars(evolutionary_examples_section=examples_section)
        else:
            # No examples available yet
            base_prompt.add_vars(evolutionary_examples_section="")
        
        return base_prompt
    
    def create_dynamic_evolutionary_prompt_function(self, 
                                                  var_names: List[str], 
                                                  var_descs: List[str], 
                                                  problem_name: str,
                                                  num_examples: int = 3,
                                                  temperature: float = 1.0,
                                                  sampling_strategy: str = "boltzmann") -> callable:
        """
        Create a dynamic evolutionary prompt function.
        
        Args:
            var_names: List of variable names
            var_descs: List of variable descriptions
            problem_name: Name of the problem
            num_examples: Number of examples to include
            temperature: Temperature for Boltzmann sampling
            sampling_strategy: Sampling strategy
            
        Returns:
            Function that takes iteration, history and returns a Prompt
        """
        def dynamic_evolutionary_prompt_fn(iteration: int, history: Dict[str, List[Any]]) -> Prompt:
            # Add previous iteration's result to buffer if it exists
            if iteration > 1 and history.get("outputs") and history.get("scores"):
                prev_output = history["outputs"][-1]  # Previous iteration's output
                prev_scores = history["scores"][-1]   # Previous iteration's scores
                
                if isinstance(prev_scores, dict) and "nmse" in prev_scores:
                    prev_nmse = prev_scores["nmse"]
                    
                    from .generation import LLMSRGeneration
                    generator = LLMSRGeneration()
                    prev_code = generator.parse_equation_code(prev_output)
                    if isinstance(prev_code, list):
                        prev_code = prev_code[0]
                        prev_code = "```python\n" + prev_code + "\n```"
                    else:
                        prev_code = prev_output

                    # breakpoint()

                    # Add to evolutionary buffer
                    self.add_program_to_buffer(
                        code=prev_code,
                        nmse_score=prev_nmse,
                        iteration=iteration-1,
                        metadata={
                            "problem_name": problem_name,
                            "var_names": var_names,
                            "var_descs": var_descs
                        }
                    )
            
            # Get current NMSE score
            current_nmse = float('inf')
            if history.get("scores") and len(history["scores"]) > 0:
                current_scores = history["scores"][-1]
                if isinstance(current_scores, dict) and "nmse" in current_scores:
                    current_nmse = current_scores["nmse"]
            
            # Create evolutionary prompt
            return self.create_evolutionary_prompt(
                var_names=var_names,
                var_descs=var_descs,
                problem_name=problem_name,
                current_iteration=iteration,
                current_nmse=current_nmse,
                num_examples=num_examples,
                temperature=temperature,
                sampling_strategy=sampling_strategy
            )
        
        return dynamic_evolutionary_prompt_fn
    
    def add_program_to_buffer(self, 
                            code: str, 
                            nmse_score: float, 
                            iteration: int,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a program to the evolutionary buffer.
        
        Args:
            code: The equation code
            nmse_score: NMSE score (lower is better)
            iteration: Current iteration number
            metadata: Additional metadata
            
        Returns:
            True if this program improved the global best score
        """
        return self.evolutionary_buffer.add_program(
            code=code,
            nmse_score=nmse_score,
            iteration=iteration,
            metadata=metadata
        )
    
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get statistics about the evolutionary buffer."""
        return self.evolutionary_buffer.get_statistics()
    
    def clear_buffer(self):
        """Clear the evolutionary buffer."""
        self.evolutionary_buffer.clear()
        self.best_nmse_history = float('inf')
    
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
