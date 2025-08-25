"""
Configuration management for the Spectrum Elucidator Toolkit.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging


@dataclass
class LLMConfig:
    """Configuration for LLM interface."""
    
    api_key: str = ""
    model: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 60


@dataclass
class ElucidationConfig:
    """Configuration for elucidation process."""
    
    max_iterations: int = 10
    similarity_threshold: float = 0.8
    temperature: float = 0.7
    save_intermediate_results: bool = True
    output_dir: str = "elucidation_results"
    log_level: str = "INFO"
    delay_between_iterations: float = 1.0
    use_nmr_predictor: bool = True
    nmr_tolerance: float = 0.20
    prefer_c_nmr: bool = True


@dataclass
class NMRPredictorConfig:
    """Configuration for NMR prediction."""
    
    use_web_scraping: bool = True
    use_llm_fallback: bool = True
    headless_browser: bool = True
    web_timeout: int = 10
    retry_attempts: int = 3
    delay_between_requests: float = 2.0


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    data_path: str = "data/updated_table.csv"
    cache_results: bool = True
    cache_dir: str = "cache"
    max_cache_size: int = 1000


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    
    style: str = "default"
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    show_plots: bool = True


@dataclass
class ToolkitConfig:
    """Main configuration for the toolkit."""
    
    llm: LLMConfig = LLMConfig()
    elucidation: ElucidationConfig = ElucidationConfig()
    nmr_predictor: NMRPredictorConfig = NMRPredictorConfig()
    data: DataConfig = DataConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure output directories exist
        Path(self.elucidation.output_dir).mkdir(exist_ok=True)
        Path(self.data.cache_dir).mkdir(exist_ok=True)


class ConfigManager:
    """Manage configuration loading, saving, and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or "config.json"
        self.config = ToolkitConfig()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration if file exists
        if os.path.exists(self.config_path):
            self.load_config()
        else:
            self.create_default_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration with loaded data
            self._update_config_from_dict(config_data)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration (optional)
        """
        save_path = config_path or self.config_path
        
        try:
            # Convert dataclass to dictionary
            config_dict = asdict(self.config)
            
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def create_default_config(self) -> None:
        """Create and save default configuration."""
        self.logger.info("Creating default configuration")
        self.save_config()
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        # Update LLM config
        if 'llm' in config_dict:
            for key, value in config_dict['llm'].items():
                if hasattr(self.config.llm, key):
                    setattr(self.config.llm, key, value)
        
        # Update elucidation config
        if 'elucidation' in config_dict:
            for key, value in config_dict['elucidation'].items():
                if hasattr(self.config.elucidation, key):
                    setattr(self.config.elucidation, key, value)
        
        # Update NMR predictor config
        if 'nmr_predictor' in config_dict:
            for key, value in config_dict['nmr_predictor'].items():
                if hasattr(self.config.nmr_predictor, key):
                    setattr(self.config.nmr_predictor, key, value)
        
        # Update data config
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(self.config.data, key):
                    setattr(self.config.data, key, value)
        
        # Update visualization config
        if 'visualization' in config_dict:
            for key, value in config_dict['visualization'].items():
                if hasattr(self.config.visualization, key):
                    setattr(self.config.visualization, key, value)
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.config.llm
    
    def get_elucidation_config(self) -> ElucidationConfig:
        """Get elucidation configuration."""
        return self.config.elucidation
    
    def get_nmr_predictor_config(self) -> NMRPredictorConfig:
        """Get NMR predictor configuration."""
        return self.config.nmr_predictor
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self.config.data
    
    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration."""
        return self.config.visualization
    
    def update_llm_config(self, **kwargs) -> None:
        """Update LLM configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.llm, key):
                setattr(self.config.llm, key, value)
                self.logger.info(f"Updated LLM config: {key} = {value}")
    
    def update_elucidation_config(self, **kwargs) -> None:
        """Update elucidation configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.elucidation, key):
                setattr(self.config.elucidation, key, value)
                self.logger.info(f"Updated elucidation config: {key} = {value}")
    
    def update_nmr_predictor_config(self, **kwargs) -> None:
        """Update NMR predictor configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.nmr_predictor, key):
                setattr(self.config.nmr_predictor, key, value)
                self.logger.info(f"Updated NMR predictor config: {key} = {value}")
    
    def update_data_config(self, **kwargs) -> None:
        """Update data configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.data, key):
                setattr(self.config.data, key, value)
                self.logger.info(f"Updated data config: {key} = {value}")
    
    def update_visualization_config(self, **kwargs) -> None:
        """Update visualization configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.visualization, key):
                setattr(self.config.visualization, key, value)
                self.logger.info(f"Updated visualization config: {key} = {value}")
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        errors = []
        
        # Check LLM config
        if not self.config.llm.api_key:
            errors.append("LLM API key is required")
        
        if self.config.llm.temperature < 0 or self.config.llm.temperature > 2:
            errors.append("LLM temperature must be between 0 and 2")
        
        # Check elucidation config
        if self.config.elucidation.max_iterations < 1:
            errors.append("Max iterations must be at least 1")
        
        if self.config.elucidation.similarity_threshold < 0 or self.config.elucidation.similarity_threshold > 1:
            errors.append("Similarity threshold must be between 0 and 1")
        
        # Check NMR predictor config
        if self.config.nmr_predictor.web_timeout < 1:
            errors.append("Web timeout must be at least 1 second")
        
        if self.config.nmr_predictor.retry_attempts < 0:
            errors.append("Retry attempts must be non-negative")
        
        # Check data config
        if not os.path.exists(self.config.data.data_path):
            errors.append(f"Data file not found: {self.config.data.data_path}")
        
        if errors:
            for error in errors:
                self.logger.error(f"Configuration error: {error}")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def get_env_config(self) -> None:
        """Load configuration from environment variables."""
        # LLM config
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.config.llm.api_key = api_key
        
        model = os.getenv('LLM_MODEL')
        if model:
            self.config.llm.model = model
        
        # Elucidation config
        max_iter = os.getenv('MAX_ITERATIONS')
        if max_iter:
            try:
                self.config.elucidation.max_iterations = int(max_iter)
            except ValueError:
                pass
        
        threshold = os.getenv('SIMILARITY_THRESHOLD')
        if threshold:
            try:
                self.config.elucidation.similarity_threshold = float(threshold)
            except ValueError:
                pass
        
        # NMR predictor config
        use_predictor = os.getenv('USE_NMR_PREDICTOR')
        if use_predictor:
            try:
                self.config.elucidation.use_nmr_predictor = use_predictor.lower() in ['true', '1', 'yes']
            except ValueError:
                pass
        
        # Data config
        data_path = os.getenv('DATA_PATH')
        if data_path:
            self.config.data.data_path = data_path
        
        self.logger.info("Environment configuration loaded")
    
    def print_config(self) -> None:
        """Print current configuration."""
        config_dict = asdict(self.config)
        print("Current Configuration:")
        print(json.dumps(config_dict, indent=2, default=str))


def create_config_template() -> str:
    """Create a configuration template."""
    template = {
        "llm": {
            "api_key": "your_openai_api_key_here",
            "model": "gpt-4",
            "max_tokens": 2000,
            "temperature": 0.7,
            "timeout": 60
        },
        "elucidation": {
            "max_iterations": 10,
            "similarity_threshold": 0.8,
            "temperature": 0.7,
            "save_intermediate_results": True,
            "output_dir": "elucidation_results",
            "log_level": "INFO",
            "delay_between_iterations": 1.0,
            "use_nmr_predictor": True,
            "nmr_tolerance": 0.20,
            "prefer_c_nmr": True
        },
        "nmr_predictor": {
            "use_web_scraping": True,
            "use_llm_fallback": True,
            "headless_browser": True,
            "web_timeout": 10,
            "retry_attempts": 3,
            "delay_between_requests": 2.0
        },
        "data": {
            "data_path": "data/updated_table.csv",
            "cache_results": True,
            "cache_dir": "cache",
            "max_cache_size": 1000
        },
        "visualization": {
            "style": "default",
            "save_plots": True,
            "plot_format": "png",
            "plot_dpi": 300,
            "show_plots": True
        }
    }
    
    return json.dumps(template, indent=2)


if __name__ == "__main__":
    # Create configuration template
    print("Configuration Template:")
    print(create_config_template())
    
    # Example usage
    config_manager = ConfigManager()
    config_manager.get_env_config()
    config_manager.print_config()
