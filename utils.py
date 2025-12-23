"""Utility functions for cave-bench."""

from typing import Callable, Dict, Any


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
            "Install it with: pip install openai-agents"
        )

    tool = function_tool(func, strict_mode=False)
    return {
        "name": func.__name__,
        "description": func.__doc__ or tool.description,
        "parameters": tool.params_json_schema
    }