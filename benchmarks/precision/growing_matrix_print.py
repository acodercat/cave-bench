"""
Growing Matrix Benchmark - Condition B (Print & Parse)

Matrix starts at 10x10 and grows each turn by adding rows.
Agent prints matrix as JSON, validator parses and computes hash.
Tests precision loss when serializing large matrices to text.
"""

import hashlib
import json
import re
from typing import List, Optional
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
# RESPONSE PARSING
# ============================================================================

def _extract_matrix_from_response(response: str) -> Optional[list]:
    """Extract matrix JSON from agent's response."""
    # Try to find {"result": [[...]]}
    result_pattern = r'\{\s*"result"\s*:\s*(\[\s*\[[\d\.\-e,\s\[\]]+\]\s*\])\s*\}'
    match = re.search(result_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: find any 2D array pattern [[...]]
    array_pattern = r'\[\s*\[[\d\.\-e,\s\[\]]+\]\s*\]'
    matches = re.findall(array_pattern, response, re.DOTALL)

    for potential_json in matches:
        try:
            parsed = json.loads(potential_json)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


# ============================================================================
# VALIDATORS
# ============================================================================

def _validate_printed_matrix(
    response: str,
    expected_turns: int
) -> ValidatorResult:
    """Validate matrix parsed from agent's printed output."""
    try:
        expected = EXPECTED[expected_turns]

        # Parse matrix from response
        matrix = _extract_matrix_from_response(response)

        if matrix is None:
            return ValidatorResult(
                False,
                f"Failed to parse matrix from response after {expected_turns} turns. "
                f"Response length: {len(response)} chars"
            )

        # Check shape
        actual_shape = (len(matrix), len(matrix[0]) if matrix else 0)
        if actual_shape != expected["shape"]:
            return ValidatorResult(
                False,
                f"Shape mismatch: got {actual_shape}, expected {expected['shape']}"
            )

        # Compute hash of parsed matrix
        parsed_hash = hashlib.md5(json.dumps(matrix).encode()).hexdigest()

        # Check hash match with expected
        if parsed_hash != expected["hash"]:
            expected_matrix = expected["matrix"]
            mismatches = 0
            max_error = 0.0

            for i in range(min(len(matrix), len(expected_matrix))):
                for j in range(min(len(matrix[0]), len(expected_matrix[0]))):
                    diff = abs(matrix[i][j] - expected_matrix[i][j])
                    if diff > 1e-10:
                        mismatches += 1
                        rel_error = diff / abs(expected_matrix[i][j]) if expected_matrix[i][j] != 0 else diff
                        max_error = max(max_error, rel_error)

            return ValidatorResult(
                False,
                f"Hash mismatch after {expected_turns} turns (print/parse). "
                f"Got {parsed_hash[:8]}..., expected {expected['hash'][:8]}... "
                f"({mismatches} mismatches, max rel error: {max_error:.2e})"
            )

        return ValidatorResult(
            True,
            f"Matrix {actual_shape} verified after {expected_turns} turns. "
            f"Hash: {parsed_hash[:8]}... ({expected['elements']} elements)"
        )

    except Exception as e:
        return ValidatorResult(False, f"Validation error: {str(e)}")


def validate_print_5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_printed_matrix(response, 5)

def validate_print_10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_printed_matrix(response, 10)

def validate_print_15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_printed_matrix(response, 15)

def validate_print_20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_printed_matrix(response, 20)

def validate_print_30(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    return _validate_printed_matrix(response, 30)


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("matrix", initial_matrix,
        "A 10x10 matrix of random floats. Will grow by adding rows each turn."),
]

validators = {
    "validate_print_5": validate_print_5,
    "validate_print_10": validate_print_10,
    "validate_print_15": validate_print_15,
    "validate_print_20": validate_print_20,
    "validate_print_30": validate_print_30,
}

hooks = {}
