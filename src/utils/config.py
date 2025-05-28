import os
import json
import yaml
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to configuration file. If None, loads default config.
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        "data_generation": {
            "model_name": "Qwen/Qwen1.5-7B-Chat",
            "temperature": 0.7,
            "max_tokens": 512,
            "domains": ["finance", "healthcare", "legal"]
        },
        "data_processing": {
            "min_quality_score": 0.6,
            "filter_invalid": True,
            "augmentation_factor": 2
        },
        "validation": {
            "min_length": 50,
            "required_elements": []
        },
        "domains": {
            "finance": {
                "prohibited_content": ["exact_predictions", "guaranteed_returns"],
                "required_entities": ["monetary_value"]
            },
            "healthcare": {
                "prohibited_content": ["medical_advice", "diagnosis"],
                "required_entities": ["medical_term"]
            },
            "legal": {
                "prohibited_content": ["legal_advice"],
                "required_entities": ["legal_term"]
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    # If no config path is provided, return default
    if not config_path:
        return default_config
    
    # If config path doesn't exist, return default
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        return default_config
    
    # Load configuration based on file extension
    try:
        _, ext = os.path.splitext(config_path)
        if ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif ext.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print(f"Warning: Unsupported config file format {ext}. Using default configuration.")
            return default_config
        
        # Merge with default config to ensure all keys exist
        merged_config = default_config.copy()
        _recursive_update(merged_config, config)
        
        return merged_config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return default_config

def _recursive_update(d: Dict[str, Any], u: Dict[str, Any]) -> None:
    """
    Recursively update a dictionary.
    
    Args:
        d: Dictionary to update
        u: Dictionary with updates
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _recursive_update(d[k], v)
        else:
            d[k] = v

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        _, ext = os.path.splitext(config_path)
        if ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif ext.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            print(f"Warning: Unsupported config file format {ext}. Using JSON format.")
            with open(f"{os.path.splitext(config_path)[0]}.json", 'w') as f:
                json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving configuration: {e}")
