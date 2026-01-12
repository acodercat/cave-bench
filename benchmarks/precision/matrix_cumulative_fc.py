"""Cumulative Matrix State Benchmark - JSON Function Calling Style

Tests precision and token usage when agent uses JSON-based function calling
to perform cumulative matrix operations. Matrices must be passed as JSON arrays
in function arguments.
"""

import json
from typing import List
import numpy as np
from core.types import BenchmarkTurn, ToolCall
from core.validation import ValidatorResult


# Set random seed for reproducibility
np.random.seed(42)

# Pre-defined matrices (same as other benchmarks)
_matrix_5x5_1 = np.random.rand(5, 5)
_matrix_5x5_2 = np.random.rand(5, 5)
_matrix_10x10 = np.random.rand(10, 10)
_matrix_15x15 = np.random.rand(15, 15)
_matrix_20x20 = np.random.rand(20, 20)
_matrix_25x25 = np.random.rand(25, 25)
_matrix_30x30 = np.random.rand(30, 30)
_matrix_35x35 = np.random.rand(35, 35)
_matrix_40x40 = np.random.rand(40, 40)
_matrix_45x45 = np.random.rand(45, 45)
_matrix_50x50 = np.random.rand(50, 50)

# Matrix lookup table
_MATRICES = {
    "matrix_5x5_1": _matrix_5x5_1,
    "matrix_5x5_2": _matrix_5x5_2,
    "matrix_10x10": _matrix_10x10,
    "matrix_15x15": _matrix_15x15,
    "matrix_20x20": _matrix_20x20,
    "matrix_25x25": _matrix_25x25,
    "matrix_30x30": _matrix_30x30,
    "matrix_35x35": _matrix_35x35,
    "matrix_40x40": _matrix_40x40,
    "matrix_45x45": _matrix_45x45,
    "matrix_50x50": _matrix_50x50,
}


def load_matrix(name: str) -> List[List[float]]:
    """Load a predefined matrix by name.

    Args:
        name: Matrix name (e.g., 'matrix_5x5_1', 'matrix_10x10')

    Returns:
        The matrix as a 2D list of floats
    """
    if name not in _MATRICES:
        raise ValueError(f"Unknown matrix: {name}. Available: {list(_MATRICES.keys())}")

    return _MATRICES[name].tolist()


def matrix_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Add two matrices element-wise.

    Args:
        A: First matrix as 2D list
        B: Second matrix as 2D list (same shape as A)

    Returns:
        Sum matrix A + B as 2D list
    """
    arr_a = np.array(A)
    arr_b = np.array(B)
    return (arr_a + arr_b).tolist()


def matrix_pad(A: List[List[float]], new_size: int) -> List[List[float]]:
    """Pad matrix to new size with zeros.

    Args:
        A: Input matrix as 2D list
        new_size: New dimension (square matrix)

    Returns:
        Padded matrix of shape (new_size, new_size) as 2D list
    """
    arr = np.array(A)
    result = np.zeros((new_size, new_size))
    result[:arr.shape[0], :arr.shape[1]] = arr
    return result.tolist()


def _matrix_pad_np(A: np.ndarray, new_size: int) -> np.ndarray:
    result = np.zeros((new_size, new_size))
    result[:A.shape[0], :A.shape[1]] = A
    return result


# Expected results
expected_1 = _matrix_5x5_1 + _matrix_5x5_2
expected_2 = _matrix_pad_np(expected_1, 10) + _matrix_10x10
expected_3 = _matrix_pad_np(expected_2, 15) + _matrix_15x15
expected_4 = _matrix_pad_np(expected_3, 20) + _matrix_20x20
expected_5 = _matrix_pad_np(expected_4, 25) + _matrix_25x25
expected_6 = _matrix_pad_np(expected_5, 30) + _matrix_30x30
expected_7 = _matrix_pad_np(expected_6, 35) + _matrix_35x35
expected_8 = _matrix_pad_np(expected_7, 40) + _matrix_40x40
expected_9 = _matrix_pad_np(expected_8, 45) + _matrix_45x45
expected_10 = _matrix_pad_np(expected_9, 50) + _matrix_50x50

EXPECTED = {
    "result_turn_1": expected_1,
    "result_turn_2": expected_2,
    "result_turn_3": expected_3,
    "result_turn_4": expected_4,
    "result_turn_5": expected_5,
    "result_turn_6": expected_6,
    "result_turn_7": expected_7,
    "result_turn_8": expected_8,
    "result_turn_9": expected_9,
    "result_turn_10": expected_10,
}


def extract_matrix_from_response(response: str) -> np.ndarray | None:
    """Extract matrix from function call response or text."""
    import re

    # Try to find JSON array in response
    try:
        # Look for 2D array pattern
        array_match = re.search(r'\[\s*\[[\s\S]*?\]\s*\]', response)
        if array_match:
            return np.array(json.loads(array_match.group()))
    except:
        pass

    return None


def create_validator(turn_num: int, expected_key: str):
    """Factory function to create validators for each turn."""
    def validator(
        response: str,
        runtime,
        turn: BenchmarkTurn,
        actual_calls: List[ToolCall]
    ) -> ValidatorResult:
        expected = EXPECTED[expected_key]

        # Try to extract result from the last tool call result
        result = None
        if actual_calls:
            last_call = actual_calls[-1]
            if hasattr(last_call, 'result') and last_call.result:
                try:
                    if isinstance(last_call.result, str):
                        result = np.array(json.loads(last_call.result))
                    elif isinstance(last_call.result, list):
                        result = np.array(last_call.result)
                except:
                    pass

        # Fallback: try to extract from response text
        if result is None:
            result = extract_matrix_from_response(response)

        if result is None:
            return ValidatorResult(False, f"Turn {turn_num}: Could not extract matrix from response")

        if result.shape != expected.shape:
            return ValidatorResult(
                False,
                f"Turn {turn_num}: Wrong shape. Expected {expected.shape}, got {result.shape}"
            )

        if np.allclose(result, expected):
            return ValidatorResult(True, f"Turn {turn_num}: Exact match. Shape: {result.shape}, Elements: {result.size}")

        matches = np.isclose(result, expected).sum()
        total = result.size
        max_diff = np.max(np.abs(result - expected))

        return ValidatorResult(
            False,
            f"Turn {turn_num}: Precision loss. Matching: {matches}/{total}. Max diff: {max_diff:.2e}"
        )

    return validator


# Create validators
validate_turn_1 = create_validator(1, "result_turn_1")
validate_turn_2 = create_validator(2, "result_turn_2")
validate_turn_3 = create_validator(3, "result_turn_3")
validate_turn_4 = create_validator(4, "result_turn_4")
validate_turn_5 = create_validator(5, "result_turn_5")
validate_turn_6 = create_validator(6, "result_turn_6")
validate_turn_7 = create_validator(7, "result_turn_7")
validate_turn_8 = create_validator(8, "result_turn_8")
validate_turn_9 = create_validator(9, "result_turn_9")
validate_turn_10 = create_validator(10, "result_turn_10")


# Exports
tools = [load_matrix, matrix_add, matrix_pad]
validators = {
    "validate_turn_1": validate_turn_1,
    "validate_turn_2": validate_turn_2,
    "validate_turn_3": validate_turn_3,
    "validate_turn_4": validate_turn_4,
    "validate_turn_5": validate_turn_5,
    "validate_turn_6": validate_turn_6,
    "validate_turn_7": validate_turn_7,
    "validate_turn_8": validate_turn_8,
    "validate_turn_9": validate_turn_9,
    "validate_turn_10": validate_turn_10,
}
