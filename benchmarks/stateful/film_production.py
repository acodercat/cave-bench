"""
Film Production - Multi-Turn Stateful Benchmark

Tests: Managing film production progression across 20 turns.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# INITIAL STATE
# ============================================================================

name = ""
status = ""
level = 0
score = 0.0
active = False
history = []
stats = {}


# ============================================================================
# HELPERS
# ============================================================================

def _compare_values(actual, expected, tolerance=0.01):
    """Recursively compare values with float tolerance."""
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)):
            return False
        return abs(actual - expected) <= tolerance
    elif isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        if not set(expected.keys()).issubset(set(actual.keys())):
            return False
        return all(_compare_values(actual[k], expected[k], tolerance) for k in expected)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_compare_values(a, e, tolerance) for a, e in zip(actual, expected))
    elif isinstance(expected, tuple):
        return actual in expected
    else:
        return actual == expected


def _validate_state(runtime: PythonRuntime, expected: dict) -> list:
    """Helper to validate runtime state against expected values."""
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if not _compare_values(val, exp_val):
            errors.append(f"{key}={val} (expected {exp_val})")
    return errors


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "concept",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["conceived"],
        "stats": {"crew": 0, "budget": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "screenplay",
        "level": 2,
        "score": 100.0,
        "active": True,
        "history": ["conceived", "script_done"],
        "stats": {"crew": 0, "budget": 0, "writers": 2, "pages": 120}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "screenplay",
        "level": 3,
        "score": 250.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed"],
        "stats": {"crew": 5, "budget": 0, "writers": 2, "pages": 120, "director": "James Chen"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "pre_production",
        "level": 4,
        "score": 500.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit"],
        "stats": {"crew": 5, "budget": 1000000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "pre_production",
        "level": 5,
        "score": 700.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast"],
        "stats": {"crew": 5, "budget": 1500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "pre_production",
        "level": 6,
        "score": 950.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete"],
        "stats": {"crew": 25, "budget": 3500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "pre_production",
        "level": 7,
        "score": 1100.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found"],
        "stats": {"crew": 55, "budget": 3500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "filming",
        "level": 8,
        "score": 1600.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started"],
        "stats": {"crew": 105, "budget": 8500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "filming",
        "level": 9,
        "score": 1900.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done"],
        "stats": {"crew": 105, "budget": 8500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 15, "dailies": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "filming",
        "level": 10,
        "score": 2400.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence"],
        "stats": {"crew": 105, "budget": 8500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 40, "dailies": 150, "stunts": 12}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "post_production",
        "level": 11,
        "score": 4800.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped"],
        "stats": {"crew": 105, "budget": 11500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "post_production",
        "level": 12,
        "score": 5200.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped", "rough_cut"],
        "stats": {"crew": 105, "budget": 11500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12, "runtime": 150, "vfx_shots": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "post_production",
        "level": 13,
        "score": 5800.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped", "rough_cut", "vfx_started"],
        "stats": {"crew": 105, "budget": 15500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12, "runtime": 150, "vfx_shots": 200}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "post_production",
        "level": 14,
        "score": 6150.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped", "rough_cut", "vfx_started", "score_complete"],
        "stats": {"crew": 105, "budget": 15500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12, "runtime": 142, "vfx_shots": 200}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "finished_editing",
        "level": 15,
        "score": 6650.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped", "rough_cut", "vfx_started", "score_complete", "final_cut"],
        "stats": {"crew": 105, "budget": 17500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12, "runtime": 142, "vfx_shots": 500}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")

def validate_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "released",
        "level": 16,
        "score": 13300.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped", "rough_cut", "vfx_started", "score_complete", "final_cut", "premiere"],
        "stats": {"crew": 105, "budget": 17500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12, "runtime": 142, "vfx_shots": 500, "critics": 85, "audience": 90}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 16 complete")

def validate_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "released",
        "level": 17,
        "score": 14300.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped", "rough_cut", "vfx_started", "score_complete", "final_cut", "premiere", "wide_release"],
        "stats": {"crew": 105, "budget": 17500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12, "runtime": 142, "vfx_shots": 500, "critics": 85, "audience": 90, "theaters": 4000, "box_office": 50000000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 17 complete")

def validate_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "released",
        "level": 18,
        "score": 16300.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped", "rough_cut", "vfx_started", "score_complete", "final_cut", "premiere", "wide_release", "box_office_hit"],
        "stats": {"crew": 105, "budget": 17500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12, "runtime": 142, "vfx_shots": 500, "critics": 90, "audience": 90, "theaters": 4000, "box_office": 150000000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 18 complete")

def validate_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn",
        "status": "awards_season",
        "level": 19,
        "score": 17800.0,
        "active": True,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped", "rough_cut", "vfx_started", "score_complete", "final_cut", "premiere", "wide_release", "box_office_hit", "nominations"],
        "stats": {"crew": 105, "budget": 17500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12, "runtime": 142, "vfx_shots": 500, "critics": 90, "audience": 90, "theaters": 4000, "box_office": 150000000, "nominations": 8, "golden_globes": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 19 complete")

def validate_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Stellar Dawn [OSCAR]",
        "status": "legendary",
        "level": 20,
        "score": 53400.0,
        "active": False,
        "history": ["conceived", "script_done", "director_signed", "greenlit", "lead_cast", "cast_complete", "locations_found", "filming_started", "week_1_done", "action_sequence", "filming_wrapped", "rough_cut", "vfx_started", "score_complete", "final_cut", "premiere", "wide_release", "box_office_hit", "nominations", "oscar_winner"],
        "stats": {"crew": 105, "budget": 17500000, "writers": 2, "pages": 120, "director": "James Chen", "studio": "Paramount", "cast": 9, "locations": 5, "scenes": 200, "dailies": 150, "stunts": 12, "runtime": 142, "vfx_shots": 500, "critics": 90, "audience": 90, "theaters": 4000, "box_office": 500000000, "nominations": 8, "golden_globes": 2, "oscars": 4}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 20 complete")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("name", name, "Entity name/identifier (str, initial: ''). Use exact quoted name from query."),
    Variable("status", status, "Current status/phase as a single lowercase word (str, initial: ''). Examples: 'concept', 'screenplay', 'pre_production', 'filming', 'post_production', 'finished_editing', 'released', 'awards_season', 'legendary'."),
    Variable("level", level, "Current level (int, initial: 0)."),
    Variable("score", score, "Cumulative score/progress (float, initial: 0.0)."),
    Variable("active", active, "Whether entity is still active (bool, initial: False). Set to False when complete."),
    Variable("history", history, "List of event identifiers (list[str], initial: []). Append one snake_case event per turn."),
    Variable("stats", stats, """A flat dictionary for tracking statistics (initial: {}). Valid keys: crew, budget, writers, pages, director, studio, cast, locations, scenes, dailies, stunts, runtime, vfx_shots, critics, audience, theaters, box_office, nominations, golden_globes, oscars. Values are numbers (int/float) or strings. Do NOT create nested dicts or add keys not in this list."""),
]

validators = {
    "validate_t1": validate_t1,
    "validate_t2": validate_t2,
    "validate_t3": validate_t3,
    "validate_t4": validate_t4,
    "validate_t5": validate_t5,
    "validate_t6": validate_t6,
    "validate_t7": validate_t7,
    "validate_t8": validate_t8,
    "validate_t9": validate_t9,
    "validate_t10": validate_t10,
    "validate_t11": validate_t11,
    "validate_t12": validate_t12,
    "validate_t13": validate_t13,
    "validate_t14": validate_t14,
    "validate_t15": validate_t15,
    "validate_t16": validate_t16,
    "validate_t17": validate_t17,
    "validate_t18": validate_t18,
    "validate_t19": validate_t19,
    "validate_t20": validate_t20,
}
