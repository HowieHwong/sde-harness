"""
X-Ray Transient Classification

A benchmark project for evaluating LLM physics reasoning through
X-ray transient source classification.
"""

__version__ = '0.1.0'

from . import utils
from . import oracles
from . import modes
from . import prompts

__all__ = ['utils', 'oracles', 'modes', 'prompts', '__version__']
