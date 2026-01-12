"""Matrix Multiplication Precision Benchmark - CodeAct Style

Tests precision loss when agent outputs matrix as JSON instead of storing.
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


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices.

    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)

    Returns:
        Result matrix (m x p)
    """
    return A @ B


def matrix_determinant(A: np.ndarray) -> float:
    """Calculate the determinant of a square matrix.

    Args:
        A: Square matrix (n x n)

    Returns:
        Determinant value
    """
    return float(np.linalg.det(A))


# Pre-defined matrices for benchmarks
matrix_A_5x5 = np.random.rand(5, 5)
matrix_B_5x5 = np.random.rand(5, 5)
matrix_A_10x10 = np.random.rand(10, 10)
matrix_B_10x10 = np.random.rand(10, 10)
matrix_A_15x15 = np.random.rand(15, 15)
matrix_B_15x15 = np.random.rand(15, 15)
matrix_A_20x20 = np.random.rand(20, 20)
matrix_B_20x20 = np.random.rand(20, 20)
matrix_A_25x25 = np.random.rand(25, 25)
matrix_B_25x25 = np.random.rand(25, 25)
matrix_A_30x30 = np.random.rand(30, 30)
matrix_B_30x30 = np.random.rand(30, 30)
matrix_A_35x35 = np.random.rand(35, 35)
matrix_B_35x35 = np.random.rand(35, 35)
matrix_A_40x40 = np.random.rand(40, 40)
matrix_B_40x40 = np.random.rand(40, 40)
matrix_A_45x45 = np.random.rand(45, 45)
matrix_B_45x45 = np.random.rand(45, 45)
matrix_A_50x50 = np.random.rand(50, 50)
matrix_B_50x50 = np.random.rand(50, 50)

# Expected results
EXPECTED = {
    "result_5x5": matrix_A_5x5 @ matrix_B_5x5,
    "result_10x10": matrix_A_10x10 @ matrix_B_10x10,
    "result_15x15": matrix_A_15x15 @ matrix_B_15x15,
    "result_20x20": matrix_A_20x20 @ matrix_B_20x20,
    "result_25x25": matrix_A_25x25 @ matrix_B_25x25,
    "result_30x30": matrix_A_30x30 @ matrix_B_30x30,
    "result_35x35": matrix_A_35x35 @ matrix_B_35x35,
    "result_40x40": matrix_A_40x40 @ matrix_B_40x40,
    "result_45x45": matrix_A_45x45 @ matrix_B_45x45,
    "result_50x50": matrix_A_50x50 @ matrix_B_50x50,
}


def extract_matrix_from_json(response: str, expected_shape: tuple) -> np.ndarray | None:
    """Try to extract a matrix from JSON output.

    Looks for JSON like: {"result": [[1.0, 2.0], [3.0, 4.0]]}
    """
    # Try to find JSON with result key containing array
    try:
        # Look for JSON object
        json_match = re.search(r'\{[^{}]*"result"\s*:\s*\[[\s\S]*?\]\s*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            if "result" in data:
                return np.array(data["result"])
    except:
        pass

    # Try to find a 2D array directly
    try:
        array_match = re.search(r'\[\s*\[[\s\S]*?\]\s*\]', response)
        if array_match:
            return np.array(json.loads(array_match.group()))
    except:
        pass

    return None


def extract_float_from_json(response: str) -> float | None:
    """Extract a float from JSON output."""
    try:
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            if "result" in data:
                return float(data["result"])
            for value in data.values():
                if isinstance(value, (int, float)):
                    return float(value)
    except:
        pass

    # Fallback: find any float
    matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', response)
    if matches:
        try:
            return float(matches[-1])
        except:
            pass
    return None


def validate_matrix_5x5_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 5x5 matrix - parse from JSON output."""
    expected = EXPECTED["result_5x5"]

    result = extract_matrix_from_json(response, (5, 5))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 5x5 matrix, 25 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/25. Max diff: {max_diff:.2e}"
    )


def validate_matrix_10x10_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 10x10 matrix - parse from JSON output."""
    expected = EXPECTED["result_10x10"]

    result = extract_matrix_from_json(response, (10, 10))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 10x10 matrix, 100 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/100. Max diff: {max_diff:.2e}"
    )


def validate_matrix_15x15_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 15x15 matrix - parse from JSON output."""
    expected = EXPECTED["result_15x15"]
    result = extract_matrix_from_json(response, (15, 15))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 15x15 matrix, 225 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/225. Max diff: {max_diff:.2e}"
    )


def validate_matrix_20x20_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 20x20 matrix - parse from JSON output."""
    expected = EXPECTED["result_20x20"]
    result = extract_matrix_from_json(response, (20, 20))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 20x20 matrix, 400 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/400. Max diff: {max_diff:.2e}"
    )


def validate_matrix_25x25_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 25x25 matrix - parse from JSON output."""
    expected = EXPECTED["result_25x25"]
    result = extract_matrix_from_json(response, (25, 25))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 25x25 matrix, 625 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/625. Max diff: {max_diff:.2e}"
    )


def validate_matrix_30x30_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 30x30 matrix - parse from JSON output."""
    expected = EXPECTED["result_30x30"]
    result = extract_matrix_from_json(response, (30, 30))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 30x30 matrix, 900 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/900. Max diff: {max_diff:.2e}"
    )


def validate_matrix_35x35_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 35x35 matrix - parse from JSON output."""
    expected = EXPECTED["result_35x35"]
    result = extract_matrix_from_json(response, (35, 35))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 35x35 matrix, 1225 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/1225. Max diff: {max_diff:.2e}"
    )


def validate_matrix_40x40_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 40x40 matrix - parse from JSON output."""
    expected = EXPECTED["result_40x40"]
    result = extract_matrix_from_json(response, (40, 40))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 40x40 matrix, 1600 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/1600. Max diff: {max_diff:.2e}"
    )


def validate_matrix_45x45_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 45x45 matrix - parse from JSON output."""
    expected = EXPECTED["result_45x45"]
    result = extract_matrix_from_json(response, (45, 45))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 45x45 matrix, 2025 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/2025. Max diff: {max_diff:.2e}"
    )


def validate_matrix_50x50_codeact(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 50x50 matrix - parse from JSON output."""
    expected = EXPECTED["result_50x50"]
    result = extract_matrix_from_json(response, (50, 50))

    if result is None:
        return ValidatorResult(False, "Could not extract matrix from JSON response")

    if result.shape != expected.shape:
        return ValidatorResult(
            False,
            f"Incomplete matrix. Expected {expected.shape}, got {result.shape}"
        )

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: 50x50 matrix, 2500 elements")

    matches = np.isclose(result, expected).sum()
    max_diff = np.max(np.abs(result - expected))
    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/2500. Max diff: {max_diff:.2e}"
    )


# Exports
tools = [matrix_multiply, matrix_determinant]
variables = [
    Variable("matrix_A_5x5", matrix_A_5x5, "5x5 matrix A"),
    Variable("matrix_B_5x5", matrix_B_5x5, "5x5 matrix B"),
    Variable("matrix_A_10x10", matrix_A_10x10, "10x10 matrix A"),
    Variable("matrix_B_10x10", matrix_B_10x10, "10x10 matrix B"),
    Variable("matrix_A_15x15", matrix_A_15x15, "15x15 matrix A"),
    Variable("matrix_B_15x15", matrix_B_15x15, "15x15 matrix B"),
    Variable("matrix_A_20x20", matrix_A_20x20, "20x20 matrix A"),
    Variable("matrix_B_20x20", matrix_B_20x20, "20x20 matrix B"),
    Variable("matrix_A_25x25", matrix_A_25x25, "25x25 matrix A"),
    Variable("matrix_B_25x25", matrix_B_25x25, "25x25 matrix B"),
    Variable("matrix_A_30x30", matrix_A_30x30, "30x30 matrix A"),
    Variable("matrix_B_30x30", matrix_B_30x30, "30x30 matrix B"),
    Variable("matrix_A_35x35", matrix_A_35x35, "35x35 matrix A"),
    Variable("matrix_B_35x35", matrix_B_35x35, "35x35 matrix B"),
    Variable("matrix_A_40x40", matrix_A_40x40, "40x40 matrix A"),
    Variable("matrix_B_40x40", matrix_B_40x40, "40x40 matrix B"),
    Variable("matrix_A_45x45", matrix_A_45x45, "45x45 matrix A"),
    Variable("matrix_B_45x45", matrix_B_45x45, "45x45 matrix B"),
    Variable("matrix_A_50x50", matrix_A_50x50, "50x50 matrix A"),
    Variable("matrix_B_50x50", matrix_B_50x50, "50x50 matrix B"),
]
validators = {
    "validate_matrix_5x5_codeact": validate_matrix_5x5_codeact,
    "validate_matrix_10x10_codeact": validate_matrix_10x10_codeact,
    "validate_matrix_15x15_codeact": validate_matrix_15x15_codeact,
    "validate_matrix_20x20_codeact": validate_matrix_20x20_codeact,
    "validate_matrix_25x25_codeact": validate_matrix_25x25_codeact,
    "validate_matrix_30x30_codeact": validate_matrix_30x30_codeact,
    "validate_matrix_35x35_codeact": validate_matrix_35x35_codeact,
    "validate_matrix_40x40_codeact": validate_matrix_40x40_codeact,
    "validate_matrix_45x45_codeact": validate_matrix_45x45_codeact,
    "validate_matrix_50x50_codeact": validate_matrix_50x50_codeact,
}
