"""
SDE-Harness: Scientific Discovery Evaluation Framework

A comprehensive, extensible framework designed to accelerate AI-powered scientific discovery.
Supports multi-provider LLMs, iterative and history-aware workflows, and advanced 
multi-round evaluation.
"""

from .core.generation import Generation
from .core.oracle import Oracle
from .core.prompt import Prompt
from .core.workflow import Workflow

__version__ = "0.1.0"
__author__ = "SDE-Harness Team"

__all__ = ["Generation", "Oracle", "Prompt", "Workflow"]