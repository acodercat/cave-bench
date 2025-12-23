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
    if set(runtime.get_variable("tags")) != {"new", "premium"}:
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
    if set(runtime.get_variable("tags")) != {"new", "premium", "verified"}:
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
    if set(tags) != {"premium", "verified"}:
        errors.append(f"tags={tags}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Profile modified")


def validate_reset_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "Bob":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 30:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 500.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"admin"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Reset initialized")

def validate_reset_double(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "BobBob":
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


# Subscription cycle validators
def validate_subscription_signup(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "charlie":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 22:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 9.99) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"monthly"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Subscription started")

def validate_subscription_upgrade(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "charlie_pro":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 22:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 29.99) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"annual", "pro"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Subscription upgraded")

def validate_subscription_cancel(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "charlie_pro":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 22:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 14.99) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != False:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"cancelled"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Subscription cancelled")


# Gaming account validators
def validate_gaming_create(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "DragonSlayer99":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 18:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 50.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"beginner", "free"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Gaming account created")

def validate_gaming_levelup(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "DragonSlayer99":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 18:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 25.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"intermediate", "free", "achievement"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Level up complete")

def validate_gaming_premium(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "DragonSlayer99_VIP":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 19:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 125.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"expert", "premium", "achievement", "vip"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Premium upgrade complete")


# Transfer scenario validators
def validate_transfer_setup(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "Dana":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 35:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 1000.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"savings", "verified"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Transfer account setup")

def validate_transfer_send(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "Dana":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 35:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 745.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"savings", "verified", "sent_money"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Transfer sent")

def validate_transfer_receive(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "Dana":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 35:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 1245.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"savings", "verified", "sent_money", "received_money"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Transfer received")


# Employee record validators
def validate_employee_hire(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "emp_eva":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 28:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 5000.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"engineering", "junior"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Employee hired")

def validate_employee_promote(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "emp_eva":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 30:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 7500.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != True:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"engineering", "senior", "team_lead"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Employee promoted")

def validate_employee_leave(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    errors = []
    if runtime.get_variable("username") != "former_eva":
        errors.append(f"username={runtime.get_variable('username')}")
    if runtime.get_variable("age") != 30:
        errors.append(f"age={runtime.get_variable('age')}")
    if abs(runtime.get_variable("balance") - 0.0) > 0.01:
        errors.append(f"balance={runtime.get_variable('balance')}")
    if runtime.get_variable("is_active") != False:
        errors.append(f"is_active={runtime.get_variable('is_active')}")
    if set(runtime.get_variable("tags")) != {"alumni"}:
        errors.append(f"tags={runtime.get_variable('tags')}")
    if errors:
        return ValidatorResult(False, "; ".join(errors))
    return ValidatorResult(True, "Employee departed")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("username", username, "User's name (str, initial: '')."),
    Variable("age", age, "User's age (int, initial: 0)."),
    Variable("balance", balance, "Account balance (float, initial: 0.0)."),
    Variable("is_active", is_active, "Active status (bool, initial: False)."),
    Variable("tags", tags, "User tags (list[str], initial: [])."),
]

validators = {
    "validate_profile_init": validate_profile_init,
    "validate_profile_update": validate_profile_update,
    "validate_profile_modify": validate_profile_modify,
    "validate_reset_init": validate_reset_init,
    "validate_reset_double": validate_reset_double,
    "validate_reset_clear": validate_reset_clear,
    "validate_subscription_signup": validate_subscription_signup,
    "validate_subscription_upgrade": validate_subscription_upgrade,
    "validate_subscription_cancel": validate_subscription_cancel,
    "validate_gaming_create": validate_gaming_create,
    "validate_gaming_levelup": validate_gaming_levelup,
    "validate_gaming_premium": validate_gaming_premium,
    "validate_transfer_setup": validate_transfer_setup,
    "validate_transfer_send": validate_transfer_send,
    "validate_transfer_receive": validate_transfer_receive,
    "validate_employee_hire": validate_employee_hire,
    "validate_employee_promote": validate_employee_promote,
    "validate_employee_leave": validate_employee_leave,
}
