"""LLMSR: LLM-based Symbolic Regression for Equation Discovery."""

from .modes.evol import LLMSREvolutionaryWorkflow, LLMSRWorkflow
from .core.generation import LLMSRGeneration
from .core.oracle import EquationOracle

__version__ = "0.2.0"
__author__ = "LLMSR Team"

__all__ = [
    'LLMSRWorkflow',
    'LLMSREvolutionaryWorkflow',
    'LLMSRGeneration',
    'EquationOracle',
]
