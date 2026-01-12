"""Cumulative Matrix State Benchmark - CodeAct Style

Tests precision loss when agent outputs growing matrix as JSON and must maintain state across turns.
Matrix grows from 5x5 to 50x50 over 10 turns.
Validator parses JSON from response text.
"""

import json
import re
from typing import List
import numpy as np
from cave_agent.python_runtime import PythonRuntime, Variable
from core.types import BenchmarkTurn, ToolCall
from core.validation import ValidatorResult


# Set random seed for reproducibility
np.random.seed(42)


def matrix_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Add two matrices element-wise.

    Args:
        A: First matrix
        B: Second matrix (same shape as A)

    Returns:
        Sum matrix: A + B
    """
    return A + B


def matrix_pad(A: np.ndarray, new_size: int) -> np.ndarray:
    """Pad matrix to new size with zeros.

    Args:
        A: Input matrix
        new_size: New dimension (square matrix)

    Returns:
        Padded matrix of shape (new_size, new_size)
    """
    result = np.zeros((new_size, new_size))
    result[:A.shape[0], :A.shape[1]] = A
    return result


# Pre-defined matrices for each turn (growing sizes: 5, 10, 15, ..., 50)
matrix_5x5_1 = np.random.rand(5, 5)
matrix_5x5_2 = np.random.rand(5, 5)
matrix_10x10 = np.random.rand(10, 10)
matrix_15x15 = np.random.rand(15, 15)
matrix_20x20 = np.random.rand(20, 20)
matrix_25x25 = np.random.rand(25, 25)
matrix_30x30 = np.random.rand(30, 30)
matrix_35x35 = np.random.rand(35, 35)
matrix_40x40 = np.random.rand(40, 40)
matrix_45x45 = np.random.rand(45, 45)
matrix_50x50 = np.random.rand(50, 50)

# Compute expected cumulative results
# Turn 1: 5x5 + 5x5 = 5x5
# Turn 2: pad(5x5 -> 10x10) + 10x10 = 10x10
# Turn 3: pad(10x10 -> 15x15) + 15x15 = 15x15
# ...
# Turn 10: pad(45x45 -> 50x50) + 50x50 = 50x50

expected_1 = matrix_5x5_1 + matrix_5x5_2  # 5x5

expected_2 = matrix_pad(expected_1, 10) + matrix_10x10  # 10x10
expected_3 = matrix_pad(expected_2, 15) + matrix_15x15  # 15x15
expected_4 = matrix_pad(expected_3, 20) + matrix_20x20  # 20x20
expected_5 = matrix_pad(expected_4, 25) + matrix_25x25  # 25x25
expected_6 = matrix_pad(expected_5, 30) + matrix_30x30  # 30x30
expected_7 = matrix_pad(expected_6, 35) + matrix_35x35  # 35x35
expected_8 = matrix_pad(expected_7, 40) + matrix_40x40  # 40x40
expected_9 = matrix_pad(expected_8, 45) + matrix_45x45  # 45x45
expected_10 = matrix_pad(expected_9, 50) + matrix_50x50  # 50x50

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


def extract_matrix_from_json(response: str) -> np.ndarray | None:
    """Try to extract a matrix from JSON output."""
    # Try JSON format: {"result": [[...], ...]}
    try:
        json_match = re.search(r'\{[^{}]*"result"\s*:\s*\[[\s\S]*?\]\s*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            if "result" in data:
                return np.array(data["result"])
    except:
        pass

    # Try 2D array directly: [[...], ...]
    try:
        array_match = re.search(r'\[\s*\[[\s\S]*?\]\s*\]', response)
        if array_match:
            return np.array(json.loads(array_match.group()))
    except:
        pass

    return None


def create_validator_codeact(turn_num: int, expected_key: str):
    """Factory function to create validators for each turn."""
    def validator(
        response: str,
        runtime: PythonRuntime,
        turn: BenchmarkTurn,
        actual_calls: List[ToolCall]
    ) -> ValidatorResult:
        expected = EXPECTED[expected_key]

        result = extract_matrix_from_json(response)

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


# Create validators for each turn
validate_turn_1_codeact = create_validator_codeact(1, "result_turn_1")
validate_turn_2_codeact = create_validator_codeact(2, "result_turn_2")
validate_turn_3_codeact = create_validator_codeact(3, "result_turn_3")
validate_turn_4_codeact = create_validator_codeact(4, "result_turn_4")
validate_turn_5_codeact = create_validator_codeact(5, "result_turn_5")
validate_turn_6_codeact = create_validator_codeact(6, "result_turn_6")
validate_turn_7_codeact = create_validator_codeact(7, "result_turn_7")
validate_turn_8_codeact = create_validator_codeact(8, "result_turn_8")
validate_turn_9_codeact = create_validator_codeact(9, "result_turn_9")
validate_turn_10_codeact = create_validator_codeact(10, "result_turn_10")


# Exports
tools = [matrix_add, matrix_pad]
variables = [
    Variable("matrix_5x5_1", matrix_5x5_1, "First 5x5 matrix for turn 1"),
    Variable("matrix_5x5_2", matrix_5x5_2, "Second 5x5 matrix for turn 1"),
    Variable("matrix_10x10", matrix_10x10, "10x10 matrix for turn 2"),
    Variable("matrix_15x15", matrix_15x15, "15x15 matrix for turn 3"),
    Variable("matrix_20x20", matrix_20x20, "20x20 matrix for turn 4"),
    Variable("matrix_25x25", matrix_25x25, "25x25 matrix for turn 5"),
    Variable("matrix_30x30", matrix_30x30, "30x30 matrix for turn 6"),
    Variable("matrix_35x35", matrix_35x35, "35x35 matrix for turn 7"),
    Variable("matrix_40x40", matrix_40x40, "40x40 matrix for turn 8"),
    Variable("matrix_45x45", matrix_45x45, "45x45 matrix for turn 9"),
    Variable("matrix_50x50", matrix_50x50, "50x50 matrix for turn 10"),
]
validators = {
    "validate_turn_1_codeact": validate_turn_1_codeact,
    "validate_turn_2_codeact": validate_turn_2_codeact,
    "validate_turn_3_codeact": validate_turn_3_codeact,
    "validate_turn_4_codeact": validate_turn_4_codeact,
    "validate_turn_5_codeact": validate_turn_5_codeact,
    "validate_turn_6_codeact": validate_turn_6_codeact,
    "validate_turn_7_codeact": validate_turn_7_codeact,
    "validate_turn_8_codeact": validate_turn_8_codeact,
    "validate_turn_9_codeact": validate_turn_9_codeact,
    "validate_turn_10_codeact": validate_turn_10_codeact,
}
