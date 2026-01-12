"""
Growing Matrix Benchmark - Condition A (Runtime Retrieve)

Matrix starts at 10x10 and grows each turn by adding rows.
Result retrieved via runtime.retrieve('matrix') - direct access.
Tests CaveAgent's ability to maintain state across turns.
"""

import hashlib
import json
from typing import List
import numpy as np
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# DATA PREPARATION
# ============================================================================

# Initial 10x5 matrix with fixed seed (5 columns to keep output smaller)
np.random.seed(42)
initial_matrix = np.random.random((10, 5)).tolist()


def _compute_expected(num_turns: int) -> dict:
    """Compute expected matrix state after num_turns of growth."""
    np.random.seed(42)
    matrix = np.random.random((10, 5)).tolist()

    for turn in range(num_turns):
        np.random.seed(42 + turn + 1)
        new_row = np.random.random(5).tolist()
        matrix.append(new_row)

    return {
        "shape": (len(matrix), len(matrix[0])),
        "elements": len(matrix) * len(matrix[0]),
        "hash": hashlib.md5(json.dumps(matrix).encode()).hexdigest(),
        "matrix": matrix,
    }


EXPECTED = {
    5: _compute_expected(5),
    10: _compute_expected(10),
    15: _compute_expected(15),
    20: _compute_expected(20),
    30: _compute_expected(30),
}


# ============================================================================
# VALIDATORS
# ============================================================================

def _validate_matrix_state(
    runtime: PythonRuntime,
    expected_turns: int
) -> ValidatorResult:
    """Validate matrix state after expected number of turns."""
    try:
        matrix = runtime.retrieve("matrix")

        if matrix is None:
            return ValidatorResult(False, "matrix variable not found")

        expected = EXPECTED[expected_turns]

        # Check shape
        actual_shape = (len(matrix), len(matrix[0]) if matrix else 0)
        if actual_shape != expected["shape"]:
            return ValidatorResult(
                False,
                f"Shape mismatch: got {actual_shape}, expected {expected['shape']}"
            )

        # Compute hash of retrieved matrix
        retrieved_hash = hashlib.md5(json.dumps(matrix).encode()).hexdigest()

        if retrieved_hash != expected["hash"]:
            expected_matrix = expected["matrix"]
            mismatches = 0
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if abs(matrix[i][j] - expected_matrix[i][j]) > 1e-10:
                        mismatches += 1

            return ValidatorResult(
                False,
                f"Hash mismatch after {expected_turns} turns. "
                f"Got {retrieved_hash[:8]}..., expected {expected['hash'][:8]}... "
                f"({mismatches} element differences)"
            )

        return ValidatorResult(
            True,
            f"Matrix {actual_shape} verified after {expected_turns} turns. "
            f"Hash: {retrieved_hash[:8]}... ({expected['elements']} elements)"
        )

    except Exception as e:
        return ValidatorResult(False, f"Validation error: {str(e)}")


def validate_retrieve_5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_matrix_state(runtime, 5)

def validate_retrieve_10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_matrix_state(runtime, 10)

def validate_retrieve_15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_matrix_state(runtime, 15)

def validate_retrieve_20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_matrix_state(runtime, 20)

def validate_retrieve_30(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_matrix_state(runtime, 30)


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("matrix", initial_matrix,
        "A 10x10 matrix of random floats. Will grow by adding rows each turn."),
]

validators = {
    "validate_retrieve_5": validate_retrieve_5,
    "validate_retrieve_10": validate_retrieve_10,
    "validate_retrieve_15": validate_retrieve_15,
    "validate_retrieve_20": validate_retrieve_20,
    "validate_retrieve_30": validate_retrieve_30,
}

hooks = {}
