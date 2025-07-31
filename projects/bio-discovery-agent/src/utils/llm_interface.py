"""LLM interface using sde-harness Generation class."""
import os
import sys
from typing import Optional, Dict, Any

# Add sde-harness to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, project_root)

from sde_harness.core import Generation


class BioLLMInterface:
    """Interface for LLM communication in BioDiscoveryAgent."""
    
    def __init__(self, model: str = "anthropic/claude-3-5-sonnet-20240620"):
        """Initialize LLM interface with specified model."""
        self.model = model
        models_file = os.path.join(project_root, "models.yaml")
        credentials_file = os.path.join(project_root, "credentials.yaml")
        
        # Initialize generation with sde-harness
        self.generator = Generation(
            models_file=models_file,
            credentials_file=credentials_file
        )
    
    def complete_text(self, prompt: str, temperature: float = 0.1, 
                     max_tokens: int = 4000, **kwargs) -> str:
        """Complete text using the configured LLM."""
        # Extract model_name from self.model (e.g., "anthropic/claude-3-5-sonnet" -> need to map to models.yaml key)
        # For now, use the full model string as model_name
        model_name = self.model
        
        gen_args = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = self.generator.generate(prompt, model_name=model_name, **gen_args)
            # Extract the text from the response
            if isinstance(response, dict):
                return response.get('content', response.get('text', str(response)))
            return str(response)
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            raise
    
    def complete_text_claude(self, prompt: str, temperature: float = 0.1,
                           max_tokens: int = 4000, **kwargs) -> str:
        """Claude-specific text completion for backward compatibility."""
        # Use Claude model
        model_name = "anthropic/claude-3-5-sonnet-20240620"
        
        gen_args = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = self.generator.generate(prompt, model_name=model_name, **gen_args)
            # Extract the text from the response
            if isinstance(response, dict):
                return response.get('content', response.get('text', str(response)))
            return str(response)
        except Exception as e:
            print(f"Error in Claude generation: {e}")
            raise


# Backward compatibility functions
def complete_text(prompt: str, model: str = "anthropic/claude-3-5-sonnet-20240620",
                 temperature: float = 0.1, max_tokens: int = 4000, **kwargs) -> str:
    """Backward compatibility wrapper for complete_text."""
    interface = BioLLMInterface(model=model)
    return interface.complete_text(prompt, temperature, max_tokens, **kwargs)


def complete_text_claude(prompt: str, temperature: float = 0.1,
                       max_tokens: int = 4000, **kwargs) -> str:
    """Backward compatibility wrapper for complete_text_claude."""
    interface = BioLLMInterface()
    return interface.complete_text_claude(prompt, temperature, max_tokens, **kwargs)