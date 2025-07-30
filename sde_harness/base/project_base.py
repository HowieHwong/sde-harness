"""
Base class for SDE-Harness projects.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..core import Generation, Oracle, Workflow


class ProjectBase(ABC):
    """
    Base class for all SDE-Harness projects.
    
    This class provides a standard interface for projects to implement
    scientific discovery workflows using the SDE-Harness framework.
    """
    
    def __init__(self, 
                 models_file: str = "config/models.yaml",
                 credentials_file: str = "config/credentials.yaml",
                 **kwargs):
        """
        Initialize the project.
        
        Args:
            models_file: Path to models configuration file
            credentials_file: Path to credentials configuration file
            **kwargs: Additional project-specific arguments
        """
        self.generator = Generation(
            models_file=models_file,
            credentials_file=credentials_file
        )
        self.oracle = Oracle()
        self.workflow = None
        self._setup_project(**kwargs)
    
    @abstractmethod
    def _setup_project(self, **kwargs):
        """
        Setup project-specific configurations.
        
        This method should be implemented by each project to initialize
        project-specific components, metrics, and workflows.
        """
        pass
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the main project workflow.
        
        Returns:
            Dict containing results of the project execution
        """
        pass
    
    def create_workflow(self, **kwargs) -> Workflow:
        """
        Create a workflow instance for this project.
        
        Returns:
            Configured Workflow instance
        """
        return Workflow(
            generator=self.generator,
            oracle=self.oracle,
            **kwargs
        )