"""
Multi-Turn (10) - Stateful Benchmark

Tests: Managing state across 10 turns with consistent variable updates.
Focus is on state persistence and correct modifications over extended interactions.
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
        val = runtime.retrieve(key)
        if not _compare_values(val, exp_val):
            errors.append(f"{key}={val} (expected {exp_val})")
    return errors


# ============================================================================
# Conversation 1: Startup Journey (10 turns)
# ============================================================================

def validate_startup_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture",
        "status": "idea",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["founded"],
        "stats": {"founders": 2, "funding": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_startup_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture",
        "status": "prototype",
        "level": 2,
        "score": 10.0,
        "active": True,
        "history": ["founded", "mvp_built"],
        "stats": {"founders": 2, "funding": 0, "users": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_startup_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture",
        "status": "seed",
        "level": 3,
        "score": 50.0,
        "active": True,
        "history": ["founded", "mvp_built", "seed_raised"],
        "stats": {"founders": 2, "funding": 500000, "users": 100, "employees": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_startup_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture",
        "status": "beta",
        "level": 4,
        "score": 100.0,
        "active": True,
        "history": ["founded", "mvp_built", "seed_raised", "beta_launch"],
        "stats": {"founders": 2, "funding": 500000, "users": 1000, "employees": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_startup_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture",
        "status": "launched",
        "level": 5,
        "score": 200.0,
        "active": True,
        "history": ["founded", "mvp_built", "seed_raised", "beta_launch", "public_launch"],
        "stats": {"founders": 2, "funding": 500000, "users": 10000, "employees": 20}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_startup_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture",
        "status": "series_a",
        "level": 6,
        "score": 500.0,
        "active": True,
        "history": ["founded", "mvp_built", "seed_raised", "beta_launch", "public_launch", "series_a"],
        "stats": {"founders": 2, "funding": 5000000, "users": 50000, "employees": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_startup_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture",
        "status": "scaling",
        "level": 7,
        "score": 1000.0,
        "active": True,
        "history": ["founded", "mvp_built", "seed_raised", "beta_launch", "public_launch", "series_a", "expansion"],
        "stats": {"founders": 2, "funding": 5000000, "users": 200000, "employees": 150, "countries": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_startup_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture",
        "status": "series_b",
        "level": 8,
        "score": 2000.0,
        "active": True,
        "history": ["founded", "mvp_built", "seed_raised", "beta_launch", "public_launch", "series_a", "expansion", "series_b"],
        "stats": {"founders": 2, "funding": 25000000, "users": 500000, "employees": 300, "countries": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_startup_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture Inc.",
        "status": "pre_ipo",
        "level": 9,
        "score": 5000.0,
        "active": True,
        "history": ["founded", "mvp_built", "seed_raised", "beta_launch", "public_launch", "series_a", "expansion", "series_b", "ipo_filing"],
        "stats": {"founders": 2, "funding": 25000000, "users": 1000000, "employees": 500, "countries": 20, "valuation": 500000000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_startup_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TechVenture Inc. [PUBLIC]",
        "status": "public",
        "level": 10,
        "score": 10000.0,
        "active": True,
        "history": ["founded", "mvp_built", "seed_raised", "beta_launch", "public_launch", "series_a", "expansion", "series_b", "ipo_filing", "ipo_complete"],
        "stats": {"founders": 2, "funding": 25000000, "users": 2000000, "employees": 1000, "countries": 30, "valuation": 1000000000, "ticker": "TVTR"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")


# ============================================================================
# Conversation 2: Student Academic Year (10 turns)
# ============================================================================

def validate_student_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan",
        "status": "enrolled",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["enrolled"],
        "stats": {"credits": 0, "gpa": 0.0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_student_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan",
        "status": "freshman",
        "level": 1,
        "score": 15.0,
        "active": True,
        "history": ["enrolled", "classes_started"],
        "stats": {"credits": 15, "gpa": 0.0, "courses": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_student_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan",
        "status": "freshman",
        "level": 1,
        "score": 45.0,
        "active": True,
        "history": ["enrolled", "classes_started", "midterms_passed"],
        "stats": {"credits": 15, "gpa": 3.2, "courses": 5, "exams": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_student_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan",
        "status": "freshman",
        "level": 1,
        "score": 90.0,
        "active": True,
        "history": ["enrolled", "classes_started", "midterms_passed", "finals_passed"],
        "stats": {"credits": 30, "gpa": 3.4, "courses": 5, "exams": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_student_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan",
        "status": "sophomore",
        "level": 2,
        "score": 100.0,
        "active": True,
        "history": ["enrolled", "classes_started", "midterms_passed", "finals_passed", "year_2_started"],
        "stats": {"credits": 30, "gpa": 3.4, "courses": 10, "exams": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_student_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan",
        "status": "sophomore",
        "level": 2,
        "score": 150.0,
        "active": True,
        "history": ["enrolled", "classes_started", "midterms_passed", "finals_passed", "year_2_started", "declared_major"],
        "stats": {"credits": 45, "gpa": 3.5, "courses": 15, "exams": 15, "major": "CS"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_student_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan",
        "status": "junior",
        "level": 3,
        "score": 200.0,
        "active": True,
        "history": ["enrolled", "classes_started", "midterms_passed", "finals_passed", "year_2_started", "declared_major", "internship"],
        "stats": {"credits": 60, "gpa": 3.6, "courses": 20, "exams": 20, "major": "CS"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_student_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan",
        "status": "junior",
        "level": 3,
        "score": 250.0,
        "active": True,
        "history": ["enrolled", "classes_started", "midterms_passed", "finals_passed", "year_2_started", "declared_major", "internship", "research_project"],
        "stats": {"credits": 75, "gpa": 3.7, "courses": 25, "exams": 25, "major": "CS", "publications": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_student_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan",
        "status": "senior",
        "level": 4,
        "score": 300.0,
        "active": True,
        "history": ["enrolled", "classes_started", "midterms_passed", "finals_passed", "year_2_started", "declared_major", "internship", "research_project", "thesis_submitted"],
        "stats": {"credits": 90, "gpa": 3.8, "courses": 30, "exams": 30, "major": "CS", "publications": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_student_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Jordan [GRADUATE]",
        "status": "graduated",
        "level": 5,
        "score": 400.0,
        "active": False,
        "history": ["enrolled", "classes_started", "midterms_passed", "finals_passed", "year_2_started", "declared_major", "internship", "research_project", "thesis_submitted", "graduated"],
        "stats": {"credits": 120, "gpa": 3.9, "courses": 40, "exams": 40, "major": "CS", "publications": 1, "honors": "magna_cum_laude"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")


# ============================================================================
# Conversation 3: Game Campaign (10 turns)
# ============================================================================

def validate_game_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero",
        "status": "tutorial",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["game_started"],
        "stats": {"hp": 100, "mp": 50, "gold": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_game_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero",
        "status": "act_1",
        "level": 5,
        "score": 500.0,
        "active": True,
        "history": ["game_started", "tutorial_complete"],
        "stats": {"hp": 150, "mp": 75, "gold": 100, "weapons": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_game_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero",
        "status": "act_1",
        "level": 10,
        "score": 1500.0,
        "active": True,
        "history": ["game_started", "tutorial_complete", "boss_1_defeated"],
        "stats": {"hp": 200, "mp": 100, "gold": 500, "weapons": 2, "bosses": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_game_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero",
        "status": "act_2",
        "level": 15,
        "score": 3000.0,
        "active": True,
        "history": ["game_started", "tutorial_complete", "boss_1_defeated", "act_2_started"],
        "stats": {"hp": 250, "mp": 125, "gold": 1000, "weapons": 3, "bosses": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_game_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero",
        "status": "act_2",
        "level": 20,
        "score": 5000.0,
        "active": True,
        "history": ["game_started", "tutorial_complete", "boss_1_defeated", "act_2_started", "boss_2_defeated"],
        "stats": {"hp": 300, "mp": 150, "gold": 2000, "weapons": 4, "bosses": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_game_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero",
        "status": "act_3",
        "level": 25,
        "score": 7500.0,
        "active": True,
        "history": ["game_started", "tutorial_complete", "boss_1_defeated", "act_2_started", "boss_2_defeated", "act_3_started"],
        "stats": {"hp": 350, "mp": 175, "gold": 3500, "weapons": 5, "bosses": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_game_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero",
        "status": "act_3",
        "level": 30,
        "score": 10000.0,
        "active": True,
        "history": ["game_started", "tutorial_complete", "boss_1_defeated", "act_2_started", "boss_2_defeated", "act_3_started", "boss_3_defeated"],
        "stats": {"hp": 400, "mp": 200, "gold": 5000, "weapons": 6, "bosses": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_game_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero",
        "status": "finale",
        "level": 35,
        "score": 15000.0,
        "active": True,
        "history": ["game_started", "tutorial_complete", "boss_1_defeated", "act_2_started", "boss_2_defeated", "act_3_started", "boss_3_defeated", "final_dungeon"],
        "stats": {"hp": 450, "mp": 225, "gold": 7500, "weapons": 7, "bosses": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_game_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero [CHAMPION]",
        "status": "victory",
        "level": 40,
        "score": 25000.0,
        "active": True,
        "history": ["game_started", "tutorial_complete", "boss_1_defeated", "act_2_started", "boss_2_defeated", "act_3_started", "boss_3_defeated", "final_dungeon", "final_boss_defeated"],
        "stats": {"hp": 500, "mp": 250, "gold": 10000, "weapons": 8, "bosses": 4}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_game_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Hero [LEGEND]",
        "status": "postgame",
        "level": 50,
        "score": 50000.0,
        "active": False,
        "history": ["game_started", "tutorial_complete", "boss_1_defeated", "act_2_started", "boss_2_defeated", "act_3_started", "boss_3_defeated", "final_dungeon", "final_boss_defeated", "100_percent"],
        "stats": {"hp": 999, "mp": 999, "gold": 99999, "weapons": 10, "bosses": 10, "secrets": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")


# ============================================================================
# Conversation 4: Fitness Journey (10 turns)
# ============================================================================

def validate_fitness_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "beginner",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["started_training"],
        "stats": {"weight": 180, "miles": 0, "sessions": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_fitness_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "beginner",
        "level": 1,
        "score": 10.0,
        "active": True,
        "history": ["started_training", "first_run"],
        "stats": {"weight": 179, "miles": 2, "sessions": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_fitness_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "novice",
        "level": 2,
        "score": 30.0,
        "active": True,
        "history": ["started_training", "first_run", "week_1_complete"],
        "stats": {"weight": 177, "miles": 10, "sessions": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_fitness_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "novice",
        "level": 2,
        "score": 60.0,
        "active": True,
        "history": ["started_training", "first_run", "week_1_complete", "first_5k"],
        "stats": {"weight": 175, "miles": 25, "sessions": 12, "personal_best": 32}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_fitness_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "intermediate",
        "level": 3,
        "score": 100.0,
        "active": True,
        "history": ["started_training", "first_run", "week_1_complete", "first_5k", "month_1_complete"],
        "stats": {"weight": 172, "miles": 50, "sessions": 20, "personal_best": 28}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_fitness_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "intermediate",
        "level": 3,
        "score": 150.0,
        "active": True,
        "history": ["started_training", "first_run", "week_1_complete", "first_5k", "month_1_complete", "first_10k"],
        "stats": {"weight": 170, "miles": 80, "sessions": 30, "personal_best": 25, "races": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_fitness_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "advanced",
        "level": 4,
        "score": 200.0,
        "active": True,
        "history": ["started_training", "first_run", "week_1_complete", "first_5k", "month_1_complete", "first_10k", "half_marathon"],
        "stats": {"weight": 168, "miles": 150, "sessions": 50, "personal_best": 23, "races": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_fitness_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "advanced",
        "level": 4,
        "score": 300.0,
        "active": True,
        "history": ["started_training", "first_run", "week_1_complete", "first_5k", "month_1_complete", "first_10k", "half_marathon", "peak_training"],
        "stats": {"weight": 165, "miles": 250, "sessions": 80, "personal_best": 21, "races": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_fitness_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "elite",
        "level": 5,
        "score": 400.0,
        "active": True,
        "history": ["started_training", "first_run", "week_1_complete", "first_5k", "month_1_complete", "first_10k", "half_marathon", "peak_training", "marathon_day"],
        "stats": {"weight": 165, "miles": 276, "sessions": 81, "personal_best": 21, "races": 6, "marathon_time": 240}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_fitness_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex [MARATHONER]",
        "status": "champion",
        "level": 6,
        "score": 500.0,
        "active": False,
        "history": ["started_training", "first_run", "week_1_complete", "first_5k", "month_1_complete", "first_10k", "half_marathon", "peak_training", "marathon_day", "goal_achieved"],
        "stats": {"weight": 165, "miles": 300, "sessions": 90, "personal_best": 20, "races": 8, "marathon_time": 225, "medals": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")


# ============================================================================
# Conversation 5: Project Development (10 turns)
# ============================================================================

def validate_project_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow",
        "status": "planning",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["project_created"],
        "stats": {"commits": 0, "tests": 0, "features": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_project_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow",
        "status": "development",
        "level": 2,
        "score": 20.0,
        "active": True,
        "history": ["project_created", "repo_initialized"],
        "stats": {"commits": 5, "tests": 0, "features": 1, "contributors": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_project_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow",
        "status": "development",
        "level": 2,
        "score": 50.0,
        "active": True,
        "history": ["project_created", "repo_initialized", "tests_added"],
        "stats": {"commits": 15, "tests": 20, "features": 3, "contributors": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_project_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow",
        "status": "alpha",
        "level": 3,
        "score": 100.0,
        "active": True,
        "history": ["project_created", "repo_initialized", "tests_added", "alpha_release"],
        "stats": {"commits": 30, "tests": 50, "features": 5, "contributors": 3, "bugs": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_project_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow",
        "status": "alpha",
        "level": 3,
        "score": 150.0,
        "active": True,
        "history": ["project_created", "repo_initialized", "tests_added", "alpha_release", "bugs_fixed"],
        "stats": {"commits": 50, "tests": 80, "features": 7, "contributors": 4, "bugs": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_project_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow",
        "status": "beta",
        "level": 4,
        "score": 200.0,
        "active": True,
        "history": ["project_created", "repo_initialized", "tests_added", "alpha_release", "bugs_fixed", "beta_release"],
        "stats": {"commits": 75, "tests": 120, "features": 10, "contributors": 5, "bugs": 0, "users": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_project_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow",
        "status": "beta",
        "level": 4,
        "score": 300.0,
        "active": True,
        "history": ["project_created", "repo_initialized", "tests_added", "alpha_release", "bugs_fixed", "beta_release", "user_feedback"],
        "stats": {"commits": 100, "tests": 150, "features": 15, "contributors": 8, "bugs": 5, "users": 200}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_project_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow",
        "status": "release_candidate",
        "level": 5,
        "score": 400.0,
        "active": True,
        "history": ["project_created", "repo_initialized", "tests_added", "alpha_release", "bugs_fixed", "beta_release", "user_feedback", "rc_release"],
        "stats": {"commits": 120, "tests": 200, "features": 18, "contributors": 10, "bugs": 0, "users": 500}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_project_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow v1.0",
        "status": "released",
        "level": 6,
        "score": 500.0,
        "active": True,
        "history": ["project_created", "repo_initialized", "tests_added", "alpha_release", "bugs_fixed", "beta_release", "user_feedback", "rc_release", "v1_released"],
        "stats": {"commits": 150, "tests": 250, "features": 20, "contributors": 12, "bugs": 0, "users": 1000, "stars": 100}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_project_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TaskFlow v1.0 [STABLE]",
        "status": "maintained",
        "level": 7,
        "score": 600.0,
        "active": False,
        "history": ["project_created", "repo_initialized", "tests_added", "alpha_release", "bugs_fixed", "beta_release", "user_feedback", "rc_release", "v1_released", "stable_milestone"],
        "stats": {"commits": 200, "tests": 300, "features": 25, "contributors": 15, "bugs": 0, "users": 5000, "stars": 500, "forks": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("name", name, "Entity name/identifier (str, initial: ''). Use exact quoted name from query."),
    Variable("status", status, "Current status/phase as a single lowercase word (str, initial: ''). Examples: 'idea', 'prototype', 'seed', 'beta', 'launched', 'tutorial', 'act_1', 'victory', 'beginner', 'novice', 'elite'. Extract the key status word, not the full phrase."),
    Variable("level", level, "Current level (int, initial: 0)."),
    Variable("score", score, "Cumulative score/progress (float, initial: 0.0)."),
    Variable("active", active, "Whether entity is still active (bool, initial: False). Set to False when complete."),
    Variable("history", history, "List of event identifiers (list[str], initial: []). Append one snake_case event per turn. Examples: 'founded', 'mvp_built', 'game_started', 'first_run'."),
    Variable("stats", stats, """Additional statistics (dict, initial: {}). Only use these exact keys based on conversation:
- startup_journey: founders, funding, users, employees, countries, valuation, ticker
- student_academic: credits, gpa, courses, exams, major, publications, honors
- game_campaign: hp, mp, gold, weapons, bosses, secrets
- fitness_journey: weight, miles, sessions, personal_best, races, marathon_time, medals
- project_development: commits, tests, features, contributors, bugs, users, stars, forks
Values are numbers (int/float) or strings. Do NOT add keys not in this list."""),
]

validators = {
    # Startup journey
    "validate_startup_t1": validate_startup_t1,
    "validate_startup_t2": validate_startup_t2,
    "validate_startup_t3": validate_startup_t3,
    "validate_startup_t4": validate_startup_t4,
    "validate_startup_t5": validate_startup_t5,
    "validate_startup_t6": validate_startup_t6,
    "validate_startup_t7": validate_startup_t7,
    "validate_startup_t8": validate_startup_t8,
    "validate_startup_t9": validate_startup_t9,
    "validate_startup_t10": validate_startup_t10,
    # Student academic year
    "validate_student_t1": validate_student_t1,
    "validate_student_t2": validate_student_t2,
    "validate_student_t3": validate_student_t3,
    "validate_student_t4": validate_student_t4,
    "validate_student_t5": validate_student_t5,
    "validate_student_t6": validate_student_t6,
    "validate_student_t7": validate_student_t7,
    "validate_student_t8": validate_student_t8,
    "validate_student_t9": validate_student_t9,
    "validate_student_t10": validate_student_t10,
    # Game campaign
    "validate_game_t1": validate_game_t1,
    "validate_game_t2": validate_game_t2,
    "validate_game_t3": validate_game_t3,
    "validate_game_t4": validate_game_t4,
    "validate_game_t5": validate_game_t5,
    "validate_game_t6": validate_game_t6,
    "validate_game_t7": validate_game_t7,
    "validate_game_t8": validate_game_t8,
    "validate_game_t9": validate_game_t9,
    "validate_game_t10": validate_game_t10,
    # Fitness journey
    "validate_fitness_t1": validate_fitness_t1,
    "validate_fitness_t2": validate_fitness_t2,
    "validate_fitness_t3": validate_fitness_t3,
    "validate_fitness_t4": validate_fitness_t4,
    "validate_fitness_t5": validate_fitness_t5,
    "validate_fitness_t6": validate_fitness_t6,
    "validate_fitness_t7": validate_fitness_t7,
    "validate_fitness_t8": validate_fitness_t8,
    "validate_fitness_t9": validate_fitness_t9,
    "validate_fitness_t10": validate_fitness_t10,
    # Project development
    "validate_project_t1": validate_project_t1,
    "validate_project_t2": validate_project_t2,
    "validate_project_t3": validate_project_t3,
    "validate_project_t4": validate_project_t4,
    "validate_project_t5": validate_project_t5,
    "validate_project_t6": validate_project_t6,
    "validate_project_t7": validate_project_t7,
    "validate_project_t8": validate_project_t8,
    "validate_project_t9": validate_project_t9,
    "validate_project_t10": validate_project_t10,
}
