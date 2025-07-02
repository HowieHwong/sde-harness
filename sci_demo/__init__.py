"""
sci_demo: A framework for iterative AI science workflows.

Usage:
    import weave
    weave.init("my_project_name")  # Initialize BEFORE importing sci_demo components
    
    from sci_demo import Generation, Workflow, Oracle, Prompt
    # Now all operations will be tracked in your weave project
"""

# Import utilities
from .utils import is_weave_initialized, safe_weave_log, safe_weave_op

# Import main classes
from .generation import Generation
from .workflow import Workflow 
from .oracle import Oracle
from .prompt import Prompt

__all__ = ['Generation', 'Workflow', 'Oracle', 'Prompt', 'is_weave_initialized', 'safe_weave_log', 'safe_weave_op']

# Framework info
# TODO: Modify the following before release+
__version__ = "0.1.0"
__author__ = "Science AI Framework Team"
__description__ = "A framework for iterative AI science workflows" 
