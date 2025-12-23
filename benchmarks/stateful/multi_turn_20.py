"""
Multi-Turn (20) - Stateful Benchmark

Tests: Managing state across 20 turns with natural language queries.
Focus is on inferring state changes from context rather than explicit instructions.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# INITIAL STATE - 7 variables for tracking state across turns
# ============================================================================

name = ""
status = ""
level = 0
score = 0.0
active = False
history = []
stats = {}


# ============================================================================
# VALIDATORS
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
        # Check that all expected keys exist and have correct values (allow extra keys)
        if not set(expected.keys()).issubset(set(actual.keys())):
            return False
        return all(_compare_values(actual[k], expected[k], tolerance) for k in expected)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_compare_values(a, e, tolerance) for a, e in zip(actual, expected))
    elif isinstance(expected, tuple):
        # Tuple means "one of these values is acceptable"
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
# Conversation 1: Space Colony (20 turns)
# ============================================================================

def validate_colony_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_colony_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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
# Conversation 2: Medical Career (20 turns)
# ============================================================================

def validate_medical_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "student",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["enrolled"],
        "stats": {"patients": 0, "procedures": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_medical_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "student",
        "level": 2,
        "score": 50.0,
        "active": True,
        "history": ["enrolled", "year_1_complete"],
        "stats": {"patients": 0, "procedures": 0, "exams": 8, "gpa": 3.5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_medical_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "clinical",
        "level": 3,
        "score": 125.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started"],
        "stats": {"patients": 0, "procedures": 0, "exams": 12, "gpa": 3.6, "rotations": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_medical_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "clinical",
        "level": 4,
        "score": 225.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation"],
        "stats": {"patients": 10, "procedures": 3, "exams": 12, "gpa": 3.6, "rotations": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_medical_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "clinical",
        "level": 5,
        "score": 325.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation"],
        "stats": {"patients": 35, "procedures": 8, "exams": 12, "gpa": 3.6, "rotations": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_medical_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "intern",
        "level": 6,
        "score": 650.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated"],
        "stats": {"patients": 35, "procedures": 8, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_medical_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "intern",
        "level": 7,
        "score": 800.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery"],
        "stats": {"patients": 50, "procedures": 16, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_medical_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "resident",
        "level": 8,
        "score": 1000.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete"],
        "stats": {"patients": 100, "procedures": 36, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 4}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_medical_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "resident",
        "level": 9,
        "score": 1200.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1"],
        "stats": {"patients": 200, "procedures": 66, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 14}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_medical_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "resident",
        "level": 10,
        "score": 1450.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2"],
        "stats": {"patients": 350, "procedures": 106, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 29}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_medical_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "resident",
        "level": 11,
        "score": 1750.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon"],
        "stats": {"patients": 380, "procedures": 126, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 37, "awards": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_medical_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "attending",
        "level": 12,
        "score": 3500.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon", "residency_complete"],
        "stats": {"patients": 380, "procedures": 126, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 37, "awards": 0, "board_certified": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_medical_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "attending",
        "level": 13,
        "score": 3900.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon", "residency_complete", "first_publication"],
        "stats": {"patients": 430, "procedures": 151, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 49, "awards": 0, "board_certified": 1, "publications": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_medical_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "attending",
        "level": 14,
        "score": 4200.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon", "residency_complete", "first_publication", "teaching_award"],
        "stats": {"patients": 430, "procedures": 151, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 49, "awards": 1, "board_certified": 1, "publications": 1, "students": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_medical_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "senior_attending",
        "level": 15,
        "score": 4700.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon", "residency_complete", "first_publication", "teaching_award", "senior_attending"],
        "stats": {"patients": 530, "procedures": 201, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 69, "awards": 1, "board_certified": 1, "publications": 4, "students": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")

def validate_medical_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "senior_attending",
        "level": 16,
        "score": 5100.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon", "residency_complete", "first_publication", "teaching_award", "senior_attending", "department_honor"],
        "stats": {"patients": 530, "procedures": 201, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 69, "awards": 3, "board_certified": 1, "publications": 6, "students": 20}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 16 complete")

def validate_medical_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "senior_attending",
        "level": 17,
        "score": 10200.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon", "residency_complete", "first_publication", "teaching_award", "senior_attending", "department_honor", "breakthrough"],
        "stats": {"patients": 730, "procedures": 301, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 207, "awards": 3, "board_certified": 1, "publications": 11, "students": 20}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 17 complete")

def validate_medical_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "department_head",
        "level": 18,
        "score": 11200.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon", "residency_complete", "first_publication", "teaching_award", "senior_attending", "department_honor", "breakthrough", "department_head"],
        "stats": {"patients": 730, "procedures": 301, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 207, "awards": 6, "board_certified": 1, "publications": 11, "students": 40}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 18 complete")

def validate_medical_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan",
        "status": "department_head",
        "level": 19,
        "score": 12700.0,
        "active": True,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon", "residency_complete", "first_publication", "teaching_award", "senior_attending", "department_honor", "breakthrough", "department_head", "national_recognition"],
        "stats": {"patients": 1230, "procedures": 501, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 257, "awards": 11, "board_certified": 1, "publications": 11, "students": 40}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 19 complete")

def validate_medical_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Dr. Morgan [CHIEF]",
        "status": "chief",
        "level": 20,
        "score": 38100.0,
        "active": False,
        "history": ["enrolled", "year_1_complete", "rotations_started", "surgery_rotation", "emergency_rotation", "graduated", "first_surgery", "internship_complete", "residency_year_1", "residency_year_2", "lead_surgeon", "residency_complete", "first_publication", "teaching_award", "senior_attending", "department_honor", "breakthrough", "department_head", "national_recognition", "chief_appointed"],
        "stats": {"patients": 2000, "procedures": 501, "exams": 16, "gpa": 3.8, "rotations": 3, "degree": 1, "surgeries": 257, "awards": 11, "board_certified": 1, "publications": 11, "students": 40, "legacy": "pioneer"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 20 complete")


# ============================================================================
# Conversation 3: E-commerce Empire (20 turns)
# ============================================================================

def validate_ecommerce_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "idea",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["conceived"],
        "stats": {"products": 0, "orders": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_ecommerce_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "launched",
        "level": 2,
        "score": 50.0,
        "active": True,
        "history": ["conceived", "website_live"],
        "stats": {"products": 10, "orders": 0, "visitors": 100}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_ecommerce_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "launched",
        "level": 3,
        "score": 150.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale"],
        "stats": {"products": 10, "orders": 1, "visitors": 100, "revenue": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_ecommerce_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "launched",
        "level": 4,
        "score": 225.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started"],
        "stats": {"products": 10, "orders": 6, "visitors": 300, "revenue": 300}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_ecommerce_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "growing",
        "level": 5,
        "score": 375.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion"],
        "stats": {"products": 30, "orders": 21, "visitors": 300, "revenue": 1050}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_ecommerce_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "growing",
        "level": 6,
        "score": 475.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire"],
        "stats": {"products": 30, "orders": 21, "visitors": 600, "revenue": 1050, "employees": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_ecommerce_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "scaling",
        "level": 7,
        "score": 950.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened"],
        "stats": {"products": 60, "orders": 71, "visitors": 600, "revenue": 3550, "employees": 1, "warehouses": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_ecommerce_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "scaling",
        "level": 8,
        "score": 1150.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team"],
        "stats": {"products": 60, "orders": 71, "visitors": 600, "revenue": 3550, "employees": 5, "warehouses": 1, "satisfaction": 4.0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_ecommerce_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "scaling",
        "level": 9,
        "score": 1450.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched"],
        "stats": {"products": 60, "orders": 171, "visitors": 1200, "revenue": 8550, "employees": 5, "warehouses": 1, "satisfaction": 4.0, "downloads": 1000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_ecommerce_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "funded",
        "level": 10,
        "score": 1950.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a"],
        "stats": {"products": 60, "orders": 171, "visitors": 1200, "revenue": 8550, "employees": 15, "warehouses": 1, "satisfaction": 4.0, "downloads": 1000, "funding": 5000000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_ecommerce_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "funded",
        "level": 11,
        "score": 2350.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international"],
        "stats": {"products": 60, "orders": 371, "visitors": 1200, "revenue": 18550, "employees": 15, "warehouses": 1, "satisfaction": 4.0, "downloads": 1000, "funding": 5000000, "countries": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_ecommerce_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "funded",
        "level": 12,
        "score": 2850.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international", "membership_launched"],
        "stats": {"products": 60, "orders": 742, "visitors": 1200, "revenue": 38550, "employees": 15, "warehouses": 1, "satisfaction": 4.0, "downloads": 1000, "funding": 5000000, "countries": 5, "members": 500}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_ecommerce_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "national",
        "level": 13,
        "score": 5700.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international", "membership_launched", "second_warehouse"],
        "stats": {"products": 60, "orders": 1042, "visitors": 1200, "revenue": 53550, "employees": 25, "warehouses": 2, "satisfaction": 4.0, "downloads": 1000, "funding": 5000000, "countries": 5, "members": 500}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_ecommerce_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "national",
        "level": 14,
        "score": 6500.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international", "membership_launched", "second_warehouse", "marketplace"],
        "stats": {"products": 180, "orders": 1542, "visitors": 1200, "revenue": 78550, "employees": 25, "warehouses": 2, "satisfaction": 4.0, "downloads": 1000, "funding": 5000000, "countries": 5, "members": 500, "sellers": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_ecommerce_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "enterprise",
        "level": 15,
        "score": 7500.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international", "membership_launched", "second_warehouse", "marketplace", "series_b"],
        "stats": {"products": 180, "orders": 1542, "visitors": 1200, "revenue": 78550, "employees": 45, "warehouses": 2, "satisfaction": 4.0, "downloads": 1000, "funding": 25000000, "countries": 5, "members": 500, "sellers": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")

def validate_ecommerce_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "enterprise",
        "level": 16,
        "score": 8300.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international", "membership_launched", "second_warehouse", "marketplace", "series_b", "logistics_expansion"],
        "stats": {"products": 180, "orders": 2542, "visitors": 1200, "revenue": 78550, "employees": 45, "warehouses": 5, "satisfaction": 4.0, "downloads": 1000, "funding": 25000000, "countries": 15, "members": 500, "sellers": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 16 complete")

def validate_ecommerce_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "enterprise",
        "level": 17,
        "score": 16600.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international", "membership_launched", "second_warehouse", "marketplace", "series_b", "logistics_expansion", "ai_deployed"],
        "stats": {"products": 180, "orders": 5084, "visitors": 2400, "revenue": 157100, "employees": 45, "warehouses": 5, "satisfaction": 4.5, "downloads": 1000, "funding": 25000000, "countries": 15, "members": 500, "sellers": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 17 complete")

def validate_ecommerce_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave",
        "status": "pre_ipo",
        "level": 18,
        "score": 18600.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international", "membership_launched", "second_warehouse", "marketplace", "series_b", "logistics_expansion", "ai_deployed", "ipo_prep"],
        "stats": {"products": 180, "orders": 5084, "visitors": 2400, "revenue": 157100, "employees": 95, "warehouses": 5, "satisfaction": 4.5, "downloads": 1000, "funding": 25000000, "countries": 15, "members": 500, "sellers": 50, "valuation": 500000000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 18 complete")

def validate_ecommerce_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave Inc.",
        "status": "public",
        "level": 19,
        "score": 23600.0,
        "active": True,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international", "membership_launched", "second_warehouse", "marketplace", "series_b", "logistics_expansion", "ai_deployed", "ipo_prep", "ipo_complete"],
        "stats": {"products": 180, "orders": 5084, "visitors": 2400, "revenue": 157100, "employees": 95, "warehouses": 5, "satisfaction": 4.5, "downloads": 1000, "funding": 25000000, "countries": 15, "members": 500, "sellers": 50, "valuation": 1000000000, "ticker": "SWAV"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 19 complete")

def validate_ecommerce_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ShopWave Inc. [GLOBAL]",
        "status": "leader",
        "level": 20,
        "score": 70800.0,
        "active": False,
        "history": ["conceived", "website_live", "first_sale", "marketing_started", "product_expansion", "first_hire", "warehouse_opened", "support_team", "app_launched", "series_a", "international", "membership_launched", "second_warehouse", "marketplace", "series_b", "logistics_expansion", "ai_deployed", "ipo_prep", "ipo_complete", "global_leader"],
        "stats": {"products": 180, "orders": 1000000, "visitors": 2400, "revenue": 157100, "employees": 95, "warehouses": 5, "satisfaction": 4.5, "downloads": 1000, "funding": 25000000, "countries": 50, "members": 500, "sellers": 10000, "valuation": 1000000000, "ticker": "SWAV"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 20 complete")


# ============================================================================
# Conversation 4: Film Production (20 turns)
# ============================================================================

def validate_film_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_film_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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
# Conversation 5: Athletic Career (20 turns)
# ============================================================================

def validate_athlete_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "amateur",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["started_training"],
        "stats": {"events": 0, "medals": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_athlete_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "amateur",
        "level": 2,
        "score": 25.0,
        "active": True,
        "history": ["started_training", "first_competition"],
        "stats": {"events": 1, "medals": 0, "pb": 10.5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_athlete_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "amateur",
        "level": 3,
        "score": 75.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal"],
        "stats": {"events": 3, "medals": 1, "pb": 10.3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_athlete_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "semi_pro",
        "level": 4,
        "score": 150.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training"],
        "stats": {"events": 3, "medals": 1, "pb": 10.3, "training_hours": 20}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_athlete_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "semi_pro",
        "level": 5,
        "score": 250.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ"],
        "stats": {"events": 6, "medals": 3, "pb": 10.1, "training_hours": 20}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_athlete_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "semi_pro",
        "level": 6,
        "score": 400.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor"],
        "stats": {"events": 6, "medals": 3, "pb": 10.1, "training_hours": 20, "sponsors": 1, "earnings": 10000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_athlete_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "professional",
        "level": 7,
        "score": 800.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team"],
        "stats": {"events": 11, "medals": 6, "pb": 9.95, "training_hours": 20, "sponsors": 1, "earnings": 10000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_athlete_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "professional",
        "level": 8,
        "score": 1000.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze"],
        "stats": {"events": 14, "medals": 7, "pb": 9.95, "training_hours": 20, "sponsors": 1, "earnings": 10000, "ranking": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_athlete_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "professional",
        "level": 9,
        "score": 1250.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified"],
        "stats": {"events": 18, "medals": 9, "pb": 9.90, "training_hours": 20, "sponsors": 1, "earnings": 10000, "ranking": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_athlete_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "elite",
        "level": 10,
        "score": 1650.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist"],
        "stats": {"events": 20, "medals": 9, "pb": 9.90, "training_hours": 20, "sponsors": 2, "earnings": 60000, "ranking": 15}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_athlete_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "elite",
        "level": 11,
        "score": 1950.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement"],
        "stats": {"events": 20, "medals": 9, "pb": 9.90, "training_hours": 20, "sponsors": 4, "earnings": 260000, "ranking": 15}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_athlete_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "elite",
        "level": 12,
        "score": 2300.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement", "record_attempt"],
        "stats": {"events": 20, "medals": 9, "pb": 9.82, "training_hours": 20, "sponsors": 4, "earnings": 260000, "ranking": 8}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_athlete_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "elite",
        "level": 13,
        "score": 4600.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement", "record_attempt", "diamond_league_win"],
        "stats": {"events": 23, "medals": 11, "pb": 9.82, "training_hours": 20, "sponsors": 4, "earnings": 360000, "ranking": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_athlete_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "world_class",
        "level": 14,
        "score": 5200.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement", "record_attempt", "diamond_league_win", "worlds_medal"],
        "stats": {"events": 25, "medals": 12, "pb": 9.82, "training_hours": 20, "sponsors": 4, "earnings": 510000, "ranking": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_athlete_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "world_class",
        "level": 15,
        "score": 5700.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement", "record_attempt", "diamond_league_win", "worlds_medal", "olympic_qualifier"],
        "stats": {"events": 28, "medals": 12, "pb": 9.78, "training_hours": 20, "sponsors": 4, "earnings": 510000, "ranking": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")

def validate_athlete_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "olympian",
        "level": 16,
        "score": 6100.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement", "record_attempt", "diamond_league_win", "worlds_medal", "olympic_qualifier", "olympic_village"],
        "stats": {"events": 28, "medals": 12, "pb": 9.78, "training_hours": 20, "sponsors": 4, "earnings": 510000, "ranking": 3, "olympics": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 16 complete")

def validate_athlete_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "olympian",
        "level": 17,
        "score": 6600.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement", "record_attempt", "diamond_league_win", "worlds_medal", "olympic_qualifier", "olympic_village", "olympic_semifinal"],
        "stats": {"events": 29, "medals": 12, "pb": 9.75, "training_hours": 20, "sponsors": 4, "earnings": 510000, "ranking": 3, "olympics": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 17 complete")

def validate_athlete_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "olympian",
        "level": 18,
        "score": 7400.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement", "record_attempt", "diamond_league_win", "worlds_medal", "olympic_qualifier", "olympic_village", "olympic_semifinal", "olympic_final"],
        "stats": {"events": 30, "medals": 12, "pb": 9.75, "training_hours": 20, "sponsors": 4, "earnings": 510000, "ranking": 2, "olympics": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 18 complete")

def validate_athlete_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift",
        "status": "olympian",
        "level": 19,
        "score": 22200.0,
        "active": True,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement", "record_attempt", "diamond_league_win", "worlds_medal", "olympic_qualifier", "olympic_village", "olympic_semifinal", "olympic_final", "olympic_gold"],
        "stats": {"events": 30, "medals": 13, "pb": 9.75, "training_hours": 20, "sponsors": 4, "earnings": 1510000, "ranking": 2, "olympics": 1, "gold": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 19 complete")

def validate_athlete_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan Swift [LEGEND]",
        "status": "legend",
        "level": 20,
        "score": 44400.0,
        "active": False,
        "history": ["started_training", "first_competition", "first_medal", "elite_training", "regional_champ", "first_sponsor", "national_team", "continental_bronze", "worlds_qualified", "worlds_finalist", "major_endorsement", "record_attempt", "diamond_league_win", "worlds_medal", "olympic_qualifier", "olympic_village", "olympic_semifinal", "olympic_final", "olympic_gold", "world_record"],
        "stats": {"events": 30, "medals": 13, "pb": 9.58, "training_hours": 20, "sponsors": 4, "earnings": 1510000, "ranking": 1, "olympics": 1, "gold": 1, "hall_of_fame": 1}
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
    Variable("status", status, "Current status/phase as a single lowercase word (str, initial: ''). Examples: 'planning', 'transit', 'landed', 'student', 'intern', 'idea', 'launched', 'concept', 'filming', 'amateur', 'professional'. Extract the key status word, not the full phrase."),
    Variable("level", level, "Current level (int, initial: 0)."),
    Variable("score", score, "Cumulative score/progress (float, initial: 0.0)."),
    Variable("active", active, "Whether entity is still active (bool, initial: False). Set to False when complete."),
    Variable("history", history, "List of event identifiers (list[str], initial: []). Append one snake_case event per turn. Examples: 'mission_approved', 'crew_selected', 'enrolled', 'first_sale'."),
    Variable("stats", stats, """A flat dictionary for tracking statistics (initial: {}). Set keys directly on stats like stats["crew"] = 5. Valid keys for each conversation type:
For space_colony: crew, supplies, scientists, engineers, budget, fuel, distance, rovers, habitats, power, water, food, minerals, discoveries, births, spacecraft, cities
For medical_career: patients, procedures, exams, gpa, rotations, degree, surgeries, awards, board_certified, publications, students, legacy
For ecommerce_empire: products, orders, visitors, revenue, employees, warehouses, satisfaction, downloads, funding, countries, members, sellers, valuation, ticker
For film_production: crew, budget, writers, pages, director, studio, cast, locations, scenes, dailies, stunts, runtime, vfx_shots, critics, audience, theaters, box_office, nominations, golden_globes, oscars
For athletic_career: events, medals, pb, training_hours, sponsors, earnings, ranking, olympics, gold, hall_of_fame
Values are numbers (int/float) or strings. Do NOT create nested dicts or add keys not in this list."""),
]

validators = {
    # Space Colony
    "validate_colony_t1": validate_colony_t1,
    "validate_colony_t2": validate_colony_t2,
    "validate_colony_t3": validate_colony_t3,
    "validate_colony_t4": validate_colony_t4,
    "validate_colony_t5": validate_colony_t5,
    "validate_colony_t6": validate_colony_t6,
    "validate_colony_t7": validate_colony_t7,
    "validate_colony_t8": validate_colony_t8,
    "validate_colony_t9": validate_colony_t9,
    "validate_colony_t10": validate_colony_t10,
    "validate_colony_t11": validate_colony_t11,
    "validate_colony_t12": validate_colony_t12,
    "validate_colony_t13": validate_colony_t13,
    "validate_colony_t14": validate_colony_t14,
    "validate_colony_t15": validate_colony_t15,
    "validate_colony_t16": validate_colony_t16,
    "validate_colony_t17": validate_colony_t17,
    "validate_colony_t18": validate_colony_t18,
    "validate_colony_t19": validate_colony_t19,
    "validate_colony_t20": validate_colony_t20,
    # Medical Career
    "validate_medical_t1": validate_medical_t1,
    "validate_medical_t2": validate_medical_t2,
    "validate_medical_t3": validate_medical_t3,
    "validate_medical_t4": validate_medical_t4,
    "validate_medical_t5": validate_medical_t5,
    "validate_medical_t6": validate_medical_t6,
    "validate_medical_t7": validate_medical_t7,
    "validate_medical_t8": validate_medical_t8,
    "validate_medical_t9": validate_medical_t9,
    "validate_medical_t10": validate_medical_t10,
    "validate_medical_t11": validate_medical_t11,
    "validate_medical_t12": validate_medical_t12,
    "validate_medical_t13": validate_medical_t13,
    "validate_medical_t14": validate_medical_t14,
    "validate_medical_t15": validate_medical_t15,
    "validate_medical_t16": validate_medical_t16,
    "validate_medical_t17": validate_medical_t17,
    "validate_medical_t18": validate_medical_t18,
    "validate_medical_t19": validate_medical_t19,
    "validate_medical_t20": validate_medical_t20,
    # E-commerce Empire
    "validate_ecommerce_t1": validate_ecommerce_t1,
    "validate_ecommerce_t2": validate_ecommerce_t2,
    "validate_ecommerce_t3": validate_ecommerce_t3,
    "validate_ecommerce_t4": validate_ecommerce_t4,
    "validate_ecommerce_t5": validate_ecommerce_t5,
    "validate_ecommerce_t6": validate_ecommerce_t6,
    "validate_ecommerce_t7": validate_ecommerce_t7,
    "validate_ecommerce_t8": validate_ecommerce_t8,
    "validate_ecommerce_t9": validate_ecommerce_t9,
    "validate_ecommerce_t10": validate_ecommerce_t10,
    "validate_ecommerce_t11": validate_ecommerce_t11,
    "validate_ecommerce_t12": validate_ecommerce_t12,
    "validate_ecommerce_t13": validate_ecommerce_t13,
    "validate_ecommerce_t14": validate_ecommerce_t14,
    "validate_ecommerce_t15": validate_ecommerce_t15,
    "validate_ecommerce_t16": validate_ecommerce_t16,
    "validate_ecommerce_t17": validate_ecommerce_t17,
    "validate_ecommerce_t18": validate_ecommerce_t18,
    "validate_ecommerce_t19": validate_ecommerce_t19,
    "validate_ecommerce_t20": validate_ecommerce_t20,
    # Film Production
    "validate_film_t1": validate_film_t1,
    "validate_film_t2": validate_film_t2,
    "validate_film_t3": validate_film_t3,
    "validate_film_t4": validate_film_t4,
    "validate_film_t5": validate_film_t5,
    "validate_film_t6": validate_film_t6,
    "validate_film_t7": validate_film_t7,
    "validate_film_t8": validate_film_t8,
    "validate_film_t9": validate_film_t9,
    "validate_film_t10": validate_film_t10,
    "validate_film_t11": validate_film_t11,
    "validate_film_t12": validate_film_t12,
    "validate_film_t13": validate_film_t13,
    "validate_film_t14": validate_film_t14,
    "validate_film_t15": validate_film_t15,
    "validate_film_t16": validate_film_t16,
    "validate_film_t17": validate_film_t17,
    "validate_film_t18": validate_film_t18,
    "validate_film_t19": validate_film_t19,
    "validate_film_t20": validate_film_t20,
    # Athletic Career
    "validate_athlete_t1": validate_athlete_t1,
    "validate_athlete_t2": validate_athlete_t2,
    "validate_athlete_t3": validate_athlete_t3,
    "validate_athlete_t4": validate_athlete_t4,
    "validate_athlete_t5": validate_athlete_t5,
    "validate_athlete_t6": validate_athlete_t6,
    "validate_athlete_t7": validate_athlete_t7,
    "validate_athlete_t8": validate_athlete_t8,
    "validate_athlete_t9": validate_athlete_t9,
    "validate_athlete_t10": validate_athlete_t10,
    "validate_athlete_t11": validate_athlete_t11,
    "validate_athlete_t12": validate_athlete_t12,
    "validate_athlete_t13": validate_athlete_t13,
    "validate_athlete_t14": validate_athlete_t14,
    "validate_athlete_t15": validate_athlete_t15,
    "validate_athlete_t16": validate_athlete_t16,
    "validate_athlete_t17": validate_athlete_t17,
    "validate_athlete_t18": validate_athlete_t18,
    "validate_athlete_t19": validate_athlete_t19,
    "validate_athlete_t20": validate_athlete_t20,
}
