"""
Prompt templates for X-ray transient classification.
"""

from .templates import (
    CLASSIFY_TEMPLATE,
    ITERATIVE_TEMPLATE,
    build_classify_prompt,
    build_iterative_prompt,
    PromptBuilder,
)

__all__ = [
    'CLASSIFY_TEMPLATE',
    'ITERATIVE_TEMPLATE',
    'build_classify_prompt',
    'build_iterative_prompt',
    'PromptBuilder',
]
