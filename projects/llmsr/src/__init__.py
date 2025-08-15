"""LLMSR: LLM-based Symbolic Regression for Equation Discovery."""

from .workflow import LLMSRWorkflow
from .workflow_sync import LLMSRWorkflowSync
from .workflow_tracked import LLMSRWorkflowTracked
from .core import LLMSRGeneration
from .oracles import EquationOracle
from .data import LLMSRDatasetLoader, EquationData
from .modes import EquationPromptTemplates

__version__ = "0.1.0"
__author__ = "LLMSR Team"

__all__ = [
    'LLMSRWorkflow',
    'LLMSRWorkflowSync',  # Synchronous version for debugging
    'LLMSRWorkflowTracked',  # Enhanced version with comprehensive tracking
    'LLMSRGeneration', 
    'EquationOracle',
    'LLMSRDatasetLoader',
    'EquationData',
    'EquationPromptTemplates'
]
