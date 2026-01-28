"""Configuration utility for loading project settings."""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, *keys, default=None):
        """Get configuration value using dot notation.
        
        Args:
            *keys: Nested keys to access
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value
    
    @property
    def data(self):
        """Get data configuration."""
        return self.config.get('data', {})
    
    @property
    def features(self):
        """Get features configuration."""
        return self.config.get('features', {})
    
    @property
    def models(self):
        """Get models configuration."""
        return self.config.get('models', {})
    
    @property
    def evaluation(self):
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    @property
    def explainability(self):
        """Get explainability configuration."""
        return self.config.get('explainability', {})
    
    @property
    def survival(self):
        """Get survival analysis configuration."""
        return self.config.get('survival', {})
    
    @property
    def causal(self):
        """Get causal inference configuration."""
        return self.config.get('causal', {})
    
    @property
    def outputs(self):
        """Get output paths configuration."""
        return self.config.get('outputs', {})


# Global config instance
config = Config()
