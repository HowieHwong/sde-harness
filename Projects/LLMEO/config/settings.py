"""
LLMEO Configuration Management

Provide configuration loading, validation, and management functions.
"""

import os
import json
from typing import Dict, Any, Optional


def get_default_config() -> Dict[str, Any]:
    """Get default config"""
    return {
        "api": {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "gemini_api_key": os.getenv("GEMINI_API_KEY"),
            "claude_api_key": os.getenv("CLAUDE_API_KEY"),
        },
        "data": {
            "ligands_file": "data/1M-space_50-ligands-full.csv",
            "fitness_file": "data/ground_truth_fitness_values.csv",
        },
        "generation": {
            "max_tokens": 5000,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "workflow": {
            "max_iterations": 2,
            "enable_multi_round_metrics": True,
            "enable_history_in_prompts": True,
        },
        "sampling": {
            "default_samples": 10,
            "default_num_samples": 10,
            "random_seed": 42,
        }
    }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load config file
    
    Args:
        config_path: config file path, if None, use default config
        
    Returns:
        config dictionary
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Warning: Failed to load config file {config_path}: {e}")
            print("Using default config")
    
    return get_default_config()


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the validity of the config
    
    Args:
        config: config dictionary
        
    Returns:
        Boolean: if the config is valid
    """
    required_files = [
        config["data"]["ligands_file"],
        config["data"]["fitness_file"]
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Data file not found {file_path}")
            return False
    
    if not config["api"]["openai_api_key"]:
        print("Warning: OpenAI API key is not set")
    
    return True


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to file
    
    Args:
        config: configuration dictionary
        config_path: save path
        
    Returns:
        Boolean: if the config is saved successfully
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error: Unable to save configuration file {config_path}: {e}")
        return False 