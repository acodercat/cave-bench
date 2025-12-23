"""
Medical Career - Multi-Turn Stateful Benchmark

Tests: Managing medical career progression across 20 turns.
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

def validate_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("name", name, "Entity name/identifier (str, initial: ''). Use exact quoted name from query."),
    Variable("status", status, "Current status/phase as a single lowercase word (str, initial: ''). Examples: 'student', 'clinical', 'intern', 'resident', 'attending', 'senior_attending', 'department_head', 'chief'."),
    Variable("level", level, "Current level (int, initial: 0)."),
    Variable("score", score, "Cumulative score/progress (float, initial: 0.0)."),
    Variable("active", active, "Whether entity is still active (bool, initial: False). Set to False when complete."),
    Variable("history", history, "List of event identifiers (list[str], initial: []). Append one snake_case event per turn."),
    Variable("stats", stats, """A flat dictionary for tracking statistics (initial: {}). Valid keys: patients, procedures, exams, gpa, rotations, degree, surgeries, awards, board_certified, publications, students, legacy. Values are numbers (int/float) or strings. Do NOT create nested dicts or add keys not in this list."""),
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
