"""Utility functions for cave-bench."""

from typing import Callable, Dict, Any
import re
from typing import Any
from json_repair import repair_json


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


def extract_json_from_response(response: str) -> Any:
    """
    Extract and parse JSON from LLM response.

    Handles markdown code blocks, raw JSON, and JSON embedded in text.
    Uses json_repair for robust parsing of malformed JSON.

    Args:
        response: Raw string response that may contain JSON in various formats

    Returns:
        Parsed JSON object (dict, list) or None if no valid JSON found

    Examples:
        >>> extract_json_from_response('{"key": "value"}')
        {'key': 'value'}

        >>> extract_json_from_response('```json\\n{"key": "value"}\\n```')
        {'key': 'value'}

        >>> extract_json_from_response('some text {"key": "value"} more')
        {'key': 'value'}

        >>> extract_json_from_response('plain text without json')
        None
    """
    response = response.strip()

    # Pattern to match JSON in markdown code blocks
    code_block_pattern = r'```(?:json|python)?\s*([\s\S]*?)\s*```'
    match = re.search(code_block_pattern, response)

    if match:
        json_str = match.group(1).strip()
    else:
        # Try to find JSON object or array in the response
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response)
        if not json_match:
            return None
        json_str = json_match.group(1)

    # Use json_repair to handle malformed JSON
    result = repair_json(json_str, return_objects=True, ensure_ascii=False)

    # Return None if result is not a dict or list
    if not isinstance(result, (dict, list)):
        return None

    return result



if __name__ == "__main__":
    response = """
    1212
    """
    print(extract_json_from_response(response))
