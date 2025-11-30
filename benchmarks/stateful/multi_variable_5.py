"""
Multi-Variable (5) - Stateful Benchmark

Tests: Managing 5 variables of diverse types modified in each turn.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# INITIAL STATE - 5 variables with diverse types
# ============================================================================

username = ""           # str
age = 0                 # int
balance = 0.0           # float
is_active = False       # bool
tags = []               # list


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_profile_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "alice":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 25:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 100.50) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if runtime.get_variable("tags") != ["new", "premium"]:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Profile initialized")

def validate_profile_update(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "alice_updated":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 26:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 150.50) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if runtime.get_variable("tags") != ["new", "premium", "verified"]:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Profile updated")

def validate_profile_modify(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "ALICE_UPDATED":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 26:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 135.45) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != False:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    tags = runtime.get_variable("tags")
    if tags != ["premium", "verified"]:
        errors.append(f"tags={tags}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Profile modified")


def validate_reset_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "bob":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 30:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 500.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if runtime.get_variable("tags") != ["admin"]:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Reset initialized")

def validate_reset_double(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "bobbob":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 60:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 1000.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if runtime.get_variable("tags") != ["admin", "admin"]:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Values doubled")

def validate_reset_clear(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 0:
        errors.append(f"age={runtime.get_variable('age')}")
    if runtime.get_variable("balance") != 0.0:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != False:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if runtime.get_variable("tags") != []:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "All cleared")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("username", username, "User's name (str, initial: '')."),
    Variable("age", age, "User's age (int, initial: 0)."),
    Variable("balance", balance, "Account balance (float, initial: 0.0)."),
    Variable("is_active", is_active, "Active status (bool, initial: False)."),
    Variable("tags", tags, "User tags (list, initial: [])."),
]

validators = {
    "validate_profile_init": validate_profile_init,
    "validate_profile_update": validate_profile_update,
    "validate_profile_modify": validate_profile_modify,
    "validate_reset_init": validate_reset_init,
    "validate_reset_double": validate_reset_double,
    "validate_reset_clear": validate_reset_clear,
}
