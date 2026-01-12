"""Matrix Multiplication Precision Benchmark

Tests whether the agent can perform matrix operations and store exact results.

CaveAgent: Retrieves the exact numpy array from runtime
CodeAct-style: Must parse from printed output (truncated, loses precision)
"""

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


def matrix_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Add two matrices element-wise.

    Args:
        A: First matrix
        B: Second matrix (same shape as A)

    Returns:
        Result matrix (same shape)
    """
    return A + B


def matrix_determinant(A: np.ndarray) -> float:
    """Calculate the determinant of a square matrix.

    Args:
        A: Square matrix (n x n)

    Returns:
        Determinant value
    """
    return float(np.linalg.det(A))


def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """Calculate the inverse of a square matrix.

    Args:
        A: Square matrix (n x n)

    Returns:
        Inverse matrix (n x n)
    """
    return np.linalg.inv(A)


# Pre-defined matrices for benchmarks (same sizes as matrix_codeact)
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

# Pre-initialized result variables
result_5x5 = np.zeros((5, 5))
result_10x10 = np.zeros((10, 10))
result_15x15 = np.zeros((15, 15))
result_20x20 = np.zeros((20, 20))
result_25x25 = np.zeros((25, 25))
result_30x30 = np.zeros((30, 30))
result_35x35 = np.zeros((35, 35))
result_40x40 = np.zeros((40, 40))
result_45x45 = np.zeros((45, 45))
result_50x50 = np.zeros((50, 50))


def validate_matrix_5x5(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 5x5 matrix multiplication (25 elements)."""
    try:
        result = runtime.retrieve("result_5x5")
    except:
        return ValidatorResult(False, "Variable 'result_5x5' not found")

    expected = EXPECTED["result_5x5"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 25 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


def validate_matrix_10x10(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 10x10 matrix multiplication (100 elements)."""
    try:
        result = runtime.retrieve("result_10x10")
    except:
        return ValidatorResult(False, "Variable 'result_10x10' not found")

    expected = EXPECTED["result_10x10"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 100 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


def validate_matrix_15x15(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 15x15 matrix multiplication (225 elements)."""
    try:
        result = runtime.retrieve("result_15x15")
    except:
        return ValidatorResult(False, "Variable 'result_15x15' not found")

    expected = EXPECTED["result_15x15"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 225 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


def validate_matrix_20x20(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 20x20 matrix multiplication (400 elements)."""
    try:
        result = runtime.retrieve("result_20x20")
    except:
        return ValidatorResult(False, "Variable 'result_20x20' not found")

    expected = EXPECTED["result_20x20"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 400 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


def validate_matrix_25x25(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 25x25 matrix multiplication (625 elements)."""
    try:
        result = runtime.retrieve("result_25x25")
    except:
        return ValidatorResult(False, "Variable 'result_25x25' not found")

    expected = EXPECTED["result_25x25"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 625 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


def validate_matrix_30x30(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 30x30 matrix multiplication (900 elements)."""
    try:
        result = runtime.retrieve("result_30x30")
    except:
        return ValidatorResult(False, "Variable 'result_30x30' not found")

    expected = EXPECTED["result_30x30"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 900 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


def validate_matrix_35x35(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 35x35 matrix multiplication (1225 elements)."""
    try:
        result = runtime.retrieve("result_35x35")
    except:
        return ValidatorResult(False, "Variable 'result_35x35' not found")

    expected = EXPECTED["result_35x35"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 1225 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


def validate_matrix_40x40(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 40x40 matrix multiplication (1600 elements)."""
    try:
        result = runtime.retrieve("result_40x40")
    except:
        return ValidatorResult(False, "Variable 'result_40x40' not found")

    expected = EXPECTED["result_40x40"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 1600 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


def validate_matrix_45x45(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 45x45 matrix multiplication (2025 elements)."""
    try:
        result = runtime.retrieve("result_45x45")
    except:
        return ValidatorResult(False, "Variable 'result_45x45' not found")

    expected = EXPECTED["result_45x45"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 2025 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


def validate_matrix_50x50(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate 50x50 matrix multiplication (2500 elements)."""
    try:
        result = runtime.retrieve("result_50x50")
    except:
        return ValidatorResult(False, "Variable 'result_50x50' not found")

    expected = EXPECTED["result_50x50"]

    if not isinstance(result, np.ndarray):
        return ValidatorResult(False, f"Wrong type. Expected ndarray, got {type(result).__name__}")

    if result.shape != expected.shape:
        return ValidatorResult(False, f"Wrong shape. Expected {expected.shape}, got {result.shape}")

    if np.allclose(result, expected):
        return ValidatorResult(True, f"Exact match: {result.shape} matrix, 2500 elements")

    matches = np.isclose(result, expected).sum()
    total = result.size
    max_diff = np.max(np.abs(result - expected))

    return ValidatorResult(
        False,
        f"Precision loss. Matching elements: {matches}/{total}. Max diff: {max_diff:.2e}"
    )


# Exports
tools = [matrix_multiply, matrix_add, matrix_determinant, matrix_inverse]
variables = [
    Variable("matrix_A_5x5", matrix_A_5x5, "5x5 matrix A for multiplication"),
    Variable("matrix_B_5x5", matrix_B_5x5, "5x5 matrix B for multiplication"),
    Variable("matrix_A_10x10", matrix_A_10x10, "10x10 matrix A for multiplication"),
    Variable("matrix_B_10x10", matrix_B_10x10, "10x10 matrix B for multiplication"),
    Variable("matrix_A_15x15", matrix_A_15x15, "15x15 matrix A for multiplication"),
    Variable("matrix_B_15x15", matrix_B_15x15, "15x15 matrix B for multiplication"),
    Variable("matrix_A_20x20", matrix_A_20x20, "20x20 matrix A for multiplication"),
    Variable("matrix_B_20x20", matrix_B_20x20, "20x20 matrix B for multiplication"),
    Variable("matrix_A_25x25", matrix_A_25x25, "25x25 matrix A for multiplication"),
    Variable("matrix_B_25x25", matrix_B_25x25, "25x25 matrix B for multiplication"),
    Variable("matrix_A_30x30", matrix_A_30x30, "30x30 matrix A for multiplication"),
    Variable("matrix_B_30x30", matrix_B_30x30, "30x30 matrix B for multiplication"),
    Variable("matrix_A_35x35", matrix_A_35x35, "35x35 matrix A for multiplication"),
    Variable("matrix_B_35x35", matrix_B_35x35, "35x35 matrix B for multiplication"),
    Variable("matrix_A_40x40", matrix_A_40x40, "40x40 matrix A for multiplication"),
    Variable("matrix_B_40x40", matrix_B_40x40, "40x40 matrix B for multiplication"),
    Variable("matrix_A_45x45", matrix_A_45x45, "45x45 matrix A for multiplication"),
    Variable("matrix_B_45x45", matrix_B_45x45, "45x45 matrix B for multiplication"),
    Variable("matrix_A_50x50", matrix_A_50x50, "50x50 matrix A for multiplication"),
    Variable("matrix_B_50x50", matrix_B_50x50, "50x50 matrix B for multiplication"),
    Variable("result_5x5", result_5x5, "Variable to store 5x5 multiplication result"),
    Variable("result_10x10", result_10x10, "Variable to store 10x10 multiplication result"),
    Variable("result_15x15", result_15x15, "Variable to store 15x15 multiplication result"),
    Variable("result_20x20", result_20x20, "Variable to store 20x20 multiplication result"),
    Variable("result_25x25", result_25x25, "Variable to store 25x25 multiplication result"),
    Variable("result_30x30", result_30x30, "Variable to store 30x30 multiplication result"),
    Variable("result_35x35", result_35x35, "Variable to store 35x35 multiplication result"),
    Variable("result_40x40", result_40x40, "Variable to store 40x40 multiplication result"),
    Variable("result_45x45", result_45x45, "Variable to store 45x45 multiplication result"),
    Variable("result_50x50", result_50x50, "Variable to store 50x50 multiplication result"),
]
validators = {
    "validate_matrix_5x5": validate_matrix_5x5,
    "validate_matrix_10x10": validate_matrix_10x10,
    "validate_matrix_15x15": validate_matrix_15x15,
    "validate_matrix_20x20": validate_matrix_20x20,
    "validate_matrix_25x25": validate_matrix_25x25,
    "validate_matrix_30x30": validate_matrix_30x30,
    "validate_matrix_35x35": validate_matrix_35x35,
    "validate_matrix_40x40": validate_matrix_40x40,
    "validate_matrix_45x45": validate_matrix_45x45,
    "validate_matrix_50x50": validate_matrix_50x50,
}
