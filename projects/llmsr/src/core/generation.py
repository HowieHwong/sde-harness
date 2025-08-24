"""LLM Generation helper tailored for equation discovery tasks.

This wrapper around `sde_harness.core.Generation` provides specialized functionality
for generating and parsing mathematical equations from LLM outputs.
"""

from __future__ import annotations

import os
import re
import ast
from typing import Optional, List, Dict, Union, Tuple
import numpy as np

from sde_harness.core import Generation


class LLMSRGeneration(Generation):
    """Specialized generation class for equation discovery tasks."""

    def __init__(self, model_name: str = "openai/gpt-4o-2024-08-06") -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        super().__init__(
            model_name=model_name,
            models_file=os.path.join(project_root, "models.yaml"),
            credentials_file=os.path.join(project_root, "credentials.yaml"),
        )

    def generate_equation(self, prompt: str, **kwargs) -> Union[str, Dict[str, str]]:
        """Generate equation code from prompt."""
        result = self.generate(prompt=prompt, **kwargs)
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        return result

    def parse_equation_code(self, response: Union[str, List[str]]) -> List[str]:
        """Extract Python equation functions from LLM output."""
        if isinstance(response, list):
            texts = response
        else:
            texts = [response]

        equations: List[str] = []
        
        for text in texts:
            # Look for Python function definitions
            # function_pattern = r'def\s+equation\s*\([^)]*\)\s*->\s*np\.ndarray\s*:.*?(?=\n\S|\Z)'
            # matches = re.findall(function_pattern, text, re.DOTALL | re.IGNORECASE)
            
            # for match in matches:
            #     # Clean up the function code
            #     cleaned = self._clean_equation_code(match)
            #     if cleaned and self._validate_equation_code(cleaned):
            #         equations.append(cleaned)
            
            # Also look for code blocks
            code_block_pattern = r'```python\s*\n(.*?)\n```'
            code_matches = re.findall(code_block_pattern, text, re.DOTALL)
            
            for code in code_matches:
                if 'def equation' in code:
                    cleaned = self._clean_equation_code(code)
                    is_valid, fixed_code = self._validate_equation_code(cleaned)
                    if is_valid:
                        equations.append(fixed_code)

        return equations

    def _clean_equation_code(self, code: str) -> str:
        """Clean and format equation code."""
        # Remove extra whitespace and normalize
        lines = code.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _validate_equation_code(self, code: str) -> tuple[bool, str]:
        """
        Basic validation of equation code.
        
        Returns:
            Tuple of (is_valid, fixed_code) where fixed_code is the validated/fixed code
        """
        try:
            # Check if it's valid Python syntax
            fixed_code = code
            try:
                ast.parse(code)
            except IndentationError:
                # Try to fix indentation by adding 4 spaces to function body lines
                import re
                lines = code.split('\n')
                new_lines = []
                in_func = False
                for line in lines:
                    if line.strip().startswith('def '):
                        in_func = True
                        new_lines.append(line)
                        continue
                    if in_func:
                        # If line is not empty and not already indented, indent it
                        if line.strip() and not line.startswith(' '):
                            new_lines.append('    ' + line)
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                fixed_code = '\n'.join(new_lines)
                ast.parse(fixed_code)
            
            # Check if it contains required elements
            has_equation_def = 'def equation' in fixed_code
            has_return = 'return' in fixed_code
            has_numpy = 'np.' in fixed_code or 'numpy.' in fixed_code
            
            is_valid = has_equation_def and has_return and has_numpy
            return is_valid, fixed_code
        except SyntaxError:
            return False, code

    def extract_equation_body(self, equation_code: str) -> str:
        """Extract just the equation body from the function."""
        # Find the line with the actual equation (usually after the docstring)
        lines = equation_code.split('\n')
        equation_line = None
        
        for line in lines:
            # Look for lines with assignment that are not function definition or docstring
            if '=' in line and not line.strip().startswith('def') and not line.strip().startswith('"""') and not line.strip().startswith('#'):
                # Remove leading whitespace and get the actual equation
                equation_line = line.strip()
                break
        
        return equation_line or ""

    def create_executable_equation(self, equation_code: str, var_names: List[str]) -> callable:
        """Create an executable function from equation code."""
        try:
            # Create a safe execution environment
            exec_globals = {
                'np': np,
                'numpy': np,
                'math': __import__('math'),
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'abs': np.abs,
                'pi': np.pi,
                'e': np.e
            }
            
            # Execute the equation code
            exec(equation_code, exec_globals)
            
            # Return the equation function
            return exec_globals['equation']
        except Exception as e:
            print(f"Error creating executable equation: {e}")
            return None


