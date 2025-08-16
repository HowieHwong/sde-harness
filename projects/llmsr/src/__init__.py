"""LLMSR: LLM-based Symbolic Regression for Equation Discovery."""

from .modes.iter import LLMSRWorkflow
from .modes.evol import LLMSREvolutionaryWorkflow
from .core.generation import LLMSRGeneration
from .core.oracle import EquationOracle
from .core.prompt import PromptTemplates, EvolutionaryPromptTemplates

__version__ = "0.1.0"
__author__ = "LLMSR Team"

__all__ = [
    'LLMSRWorkflow',
    'LLMSREvolutionaryWorkflow',
    'LLMSRGeneration', 
    'EquationOracle',
    'PromptTemplates',
    'EvolutionaryPromptTemplates'
]
