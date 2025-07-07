"""
sci_demo: A framework for iterative AI science workflows.

Usage:
    import weave
    weave.init("my_project_name")  # Initialize BEFORE importing sci_demo components
    
    from sci_demo import Generation, Workflow, Oracle, Prompt
    # Now all operations will be tracked in your weave project
"""

# Import main classes
from .generation import Generation
from .workflow import Workflow 
from .oracle import Oracle
from .prompt import Prompt

__all__ = ['Generation', 'Workflow', 'Oracle', 'Prompt']

# Framework info
from .__info__ import __version__, __author__, __description__
