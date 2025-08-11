"""
MOF Generation class - wrapper around sde_harness Generation
"""

import sys
import os
from typing import Any, Dict, List, Optional

# Add sde_harness to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from sde_harness.core import Generation


class MOFGeneration:
    """
    MOF-specific wrapper around sde_harness Generation class.
    Provides simplified interface for generating MOF names using various LLM providers.
    """
    
    def __init__(
        self,
        models_file: str = None,
        credentials_file: str = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_workers: int = 4,
    ):
        """
        Initialize MOF generation with sde_harness Generation backend.
        
        Args:
            models_file: Path to models.yaml configuration
            credentials_file: Path to credentials.yaml configuration  
            model_name: Default model to use
            device: Device for local models
            max_workers: Number of concurrent workers
        """
        # Default to sde_harness root config files if not specified
        if models_file is None:
            models_file = os.path.join(project_root, "models.yaml")
        if credentials_file is None:
            credentials_file = os.path.join(project_root, "credentials.yaml")
            
        self.generator = Generation(
            models_file=models_file,
            credentials_file=credentials_file,
            model_name=model_name,
            device=device,
            max_workers=max_workers
        )
        
    def generate_mof_names(
        self, 
        prompt: str, 
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate MOF names using the specified model.
        
        Args:
            prompt: Input prompt for MOF generation
            model_name: Model to use (overrides default)
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing generated text and metadata
        """
        return self.generator.generate(
            prompt=prompt,
            model_name=model_name,
            **kwargs
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        model_name: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate MOF names for multiple prompts concurrently.
        
        Args:
            prompts: List of input prompts
            model_name: Model to use
            **kwargs: Additional generation parameters
            
        Returns:
            List of generation results
        """
        results = []
        for prompt in prompts:
            result = self.generate_mof_names(prompt, model_name, **kwargs)
            results.append(result)
        return results
    
    def close(self):
        """Clean up resources."""
        self.generator.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()