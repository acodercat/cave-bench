"""
Space Colony - Multi-Turn Stateful Benchmark

Tests: Managing Mars colony mission state across 20 turns.
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
        "name": "Ares Prime",
        "status": "planning",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["mission_approved"],
        "stats": {"crew": 0, "supplies": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "planning",
        "level": 2,
        "score": 50.0,
        "active": True,
        "history": ["mission_approved", "crew_selected"],
        "stats": {"crew": 12, "supplies": 0, "scientists": 4}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "construction",
        "level": 3,
        "score": 150.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started"],
        "stats": {"crew": 12, "supplies": 0, "scientists": 4, "engineers": 6, "budget": 1000000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "construction",
        "level": 4,
        "score": 250.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded"],
        "stats": {"crew": 12, "supplies": 500, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 100}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "transit",
        "level": 5,
        "score": 450.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched"],
        "stats": {"crew": 12, "supplies": 500, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 80, "distance": 1000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "transit",
        "level": 6,
        "score": 600.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey"],
        "stats": {"crew": 12, "supplies": 450, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 50, "distance": 51000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "orbit",
        "level": 7,
        "score": 900.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit"],
        "stats": {"crew": 12, "supplies": 450, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 30, "distance": 100000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "orbit",
        "level": 8,
        "score": 1000.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected"],
        "stats": {"crew": 12, "supplies": 450, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 30, "distance": 100000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "landed",
        "level": 9,
        "score": 2000.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed"],
        "stats": {"crew": 12, "supplies": 450, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "base_camp",
        "level": 10,
        "score": 2500.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established"],
        "stats": {"crew": 12, "supplies": 350, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 2, "habitats": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "base_camp",
        "level": 11,
        "score": 2700.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed"],
        "stats": {"crew": 12, "supplies": 350, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 2, "habitats": 2, "power": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "expanding",
        "level": 12,
        "score": 3000.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed", "water_found"],
        "stats": {"crew": 12, "supplies": 350, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 2, "habitats": 2, "power": 100, "water": 100}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "expanding",
        "level": 13,
        "score": 3250.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed", "water_found", "greenhouse_built"],
        "stats": {"crew": 12, "supplies": 300, "scientists": 4, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 2, "habitats": 3, "power": 100, "water": 100, "food": 20}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "expanding",
        "level": 14,
        "score": 3650.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed", "water_found", "greenhouse_built", "reinforcements"],
        "stats": {"crew": 20, "supplies": 500, "scientists": 8, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 2, "habitats": 3, "power": 100, "water": 100, "food": 20}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "industrial",
        "level": 15,
        "score": 4150.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed", "water_found", "greenhouse_built", "reinforcements", "mining_started"],
        "stats": {"crew": 20, "supplies": 500, "scientists": 8, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 4, "habitats": 3, "power": 100, "water": 100, "food": 20, "minerals": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")

def validate_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "industrial",
        "level": 16,
        "score": 4550.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed", "water_found", "greenhouse_built", "reinforcements", "mining_started", "lab_complete"],
        "stats": {"crew": 20, "supplies": 500, "scientists": 8, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 4, "habitats": 5, "power": 100, "water": 100, "food": 20, "minerals": 50, "discoveries": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 16 complete")

def validate_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "industrial",
        "level": 17,
        "score": 9100.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed", "water_found", "greenhouse_built", "reinforcements", "mining_started", "lab_complete", "major_discovery"],
        "stats": {"crew": 20, "supplies": 500, "scientists": 16, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 4, "habitats": 5, "power": 100, "water": 100, "food": 20, "minerals": 50, "discoveries": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 17 complete")

def validate_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "self_sufficient",
        "level": 18,
        "score": 10100.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed", "water_found", "greenhouse_built", "reinforcements", "mining_started", "lab_complete", "major_discovery", "self_sufficient"],
        "stats": {"crew": 20, "supplies": 500, "scientists": 16, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 4, "habitats": 5, "power": 100, "water": 200, "food": 40, "minerals": 50, "discoveries": 5, "births": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 18 complete")

def validate_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime",
        "status": "hub",
        "level": 19,
        "score": 11600.0,
        "active": True,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed", "water_found", "greenhouse_built", "reinforcements", "mining_started", "lab_complete", "major_discovery", "self_sufficient", "hub_status"],
        "stats": {"crew": 40, "supplies": 500, "scientists": 16, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 4, "habitats": 15, "power": 100, "water": 200, "food": 40, "minerals": 50, "discoveries": 5, "births": 1, "spacecraft": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 19 complete")

def validate_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ares Prime [CIVILIZATION]",
        "status": "civilization",
        "level": 20,
        "score": 34800.0,
        "active": False,
        "history": ["mission_approved", "crew_selected", "rocket_started", "supplies_loaded", "launched", "mid_journey", "mars_orbit", "site_selected", "landed", "base_established", "solar_deployed", "water_found", "greenhouse_built", "reinforcements", "mining_started", "lab_complete", "major_discovery", "self_sufficient", "hub_status", "civilization_achieved"],
        "stats": {"crew": 100, "supplies": 500, "scientists": 16, "engineers": 6, "budget": 1000000, "fuel": 20, "distance": 100000, "rovers": 4, "habitats": 20, "power": 100, "water": 200, "food": 40, "minerals": 50, "discoveries": 5, "births": 1, "spacecraft": 3, "cities": 1}
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
    Variable("status", status, "Current status/phase as a single lowercase word (str, initial: ''). Examples: 'planning', 'transit', 'landed', 'orbit', 'base_camp', 'expanding', 'industrial', 'self_sufficient', 'hub', 'civilization'."),
    Variable("level", level, "Current level (int, initial: 0)."),
    Variable("score", score, "Cumulative score/progress (float, initial: 0.0)."),
    Variable("active", active, "Whether entity is still active (bool, initial: False). Set to False when complete."),
    Variable("history", history, "List of event identifiers (list[str], initial: []). Append one snake_case event per turn."),
    Variable("stats", stats, """A flat dictionary for tracking statistics (initial: {}). Valid keys: crew, supplies, scientists, engineers, budget, fuel, distance, rovers, habitats, power, water, food, minerals, discoveries, births, spacecraft, cities. Values are numbers (int/float). Do NOT create nested dicts or add keys not in this list."""),
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
