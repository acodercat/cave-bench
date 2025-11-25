"""Utility functions for configuration and model loading."""

import tomllib
from typing import TypedDict, Callable, Dict, Any, Optional
from pathlib import Path


class ModelConfig(TypedDict, total=False):
    """Type definition for model configuration."""
    model_id: str
    api_key: str
    base_url: str
    temperature: float


def load_model_config(
    model_name: str,
    config_path: str = "models.toml"
) -> ModelConfig:
    """
    Load model configuration from a TOML file.

    Args:
        model_name: The name of the model configuration to load
        config_path: Path to the TOML configuration file (default: models.toml)

    Returns:
        Dictionary containing model configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If model_name not found in config
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found. "
            f"Create it from models.toml.example"
        )

    with open(config_file, "rb") as f:
        models = tomllib.load(f)

    if model_name not in models:
        available = list(models.keys())
        raise ValueError(
            f"Model '{model_name}' not found in {config_path}. "
            f"Available models: {', '.join(available)}"
        )

    return models[model_name]


def function_to_schema(func: Callable) -> Dict[str, Any]:
    """
    Convert a Python function to an OpenAI-compatible tool schema.

    Args:
        func: Python function with type hints and docstring

    Returns:
        Dictionary with name, description, and parameters schema

    Note:
        This requires the 'agents' package. Used for compatibility
        with non-CaveAgent frameworks.
    """
    try:
        from agents import function_tool
    except ImportError:
        raise ImportError(
            "The 'agents' package is required for function_to_schema. "
            "Install it with: pip install agents"
        )

    tool = function_tool(func, strict_mode=False)
    return {
        "name": func.__name__,
        "description": func.__doc__ or tool.description,
        "parameters": tool.params_json_schema
    }