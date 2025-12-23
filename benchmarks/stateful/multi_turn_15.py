"""
Multi-Turn (15) - Stateful Benchmark

Tests: Managing state across 15 turns with consistent variable updates.
Focus is on state persistence and correct modifications over extended interactions.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# INITIAL STATE - 6 variables for tracking state across turns
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
# Conversation 1: Space Exploration Mission (15 turns)
# ============================================================================

def validate_space_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "pre_launch",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["mission_created"],
        "stats": {"crew": 6, "fuel": 100}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_space_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "launched",
        "level": 2,
        "score": 100.0,
        "active": True,
        "history": ["mission_created", "launched"],
        "stats": {"crew": 6, "fuel": 95, "altitude": 100}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_space_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "orbit",
        "level": 3,
        "score": 250.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit"],
        "stats": {"crew": 6, "fuel": 85, "altitude": 400}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_space_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "docked",
        "level": 4,
        "score": 500.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss"],
        "stats": {"crew": 6, "fuel": 80, "altitude": 400, "samples": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_space_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "eva",
        "level": 5,
        "score": 750.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk"],
        "stats": {"crew": 6, "fuel": 80, "altitude": 400, "samples": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_space_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "transit_moon",
        "level": 6,
        "score": 1000.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss"],
        "stats": {"crew": 6, "fuel": 60, "altitude": 10000, "samples": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_space_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "lunar_orbit",
        "level": 7,
        "score": 1500.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss", "lunar_orbit"],
        "stats": {"crew": 6, "fuel": 45, "altitude": 100, "samples": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_space_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "lunar_surface",
        "level": 8,
        "score": 2500.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss", "lunar_orbit", "moon_landing"],
        "stats": {"crew": 6, "fuel": 35, "altitude": 0, "samples": 5, "moonwalks": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_space_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "lunar_exploration",
        "level": 9,
        "score": 3500.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss", "lunar_orbit", "moon_landing", "first_moonwalk"],
        "stats": {"crew": 6, "fuel": 35, "altitude": 0, "samples": 20, "moonwalks": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_space_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "lunar_exploration",
        "level": 10,
        "score": 5000.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss", "lunar_orbit", "moon_landing", "first_moonwalk", "base_established"],
        "stats": {"crew": 6, "fuel": 35, "altitude": 0, "samples": 50, "moonwalks": 3, "base": True}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_space_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "lunar_ascent",
        "level": 11,
        "score": 6000.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss", "lunar_orbit", "moon_landing", "first_moonwalk", "base_established", "lunar_liftoff"],
        "stats": {"crew": 6, "fuel": 25, "altitude": 100, "samples": 50, "moonwalks": 3, "base": True}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_space_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "transit_earth",
        "level": 12,
        "score": 7500.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss", "lunar_orbit", "moon_landing", "first_moonwalk", "base_established", "lunar_liftoff", "earth_transit"],
        "stats": {"crew": 6, "fuel": 15, "altitude": 200000, "samples": 50, "moonwalks": 3, "base": True}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_space_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "reentry",
        "level": 13,
        "score": 9000.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss", "lunar_orbit", "moon_landing", "first_moonwalk", "base_established", "lunar_liftoff", "earth_transit", "reentry"],
        "stats": {"crew": 6, "fuel": 5, "altitude": 100, "samples": 50, "moonwalks": 3, "base": True}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_space_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey",
        "status": "splashdown",
        "level": 14,
        "score": 10000.0,
        "active": True,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss", "lunar_orbit", "moon_landing", "first_moonwalk", "base_established", "lunar_liftoff", "earth_transit", "reentry", "splashdown"],
        "stats": {"crew": 6, "fuel": 0, "altitude": 0, "samples": 50, "moonwalks": 3, "base": True}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_space_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Odyssey [LEGENDARY]",
        "status": "mission_complete",
        "level": 15,
        "score": 15000.0,
        "active": False,
        "history": ["mission_created", "launched", "reached_orbit", "docked_iss", "spacewalk", "departed_iss", "lunar_orbit", "moon_landing", "first_moonwalk", "base_established", "lunar_liftoff", "earth_transit", "reentry", "splashdown", "heroes_welcome"],
        "stats": {"crew": 6, "fuel": 0, "altitude": 0, "samples": 50, "moonwalks": 3, "base": True, "medals": 6}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")


# ============================================================================
# Conversation 2: Restaurant Business (15 turns)
# ============================================================================

def validate_restaurant_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "food_truck",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["started"],
        "stats": {"staff": 2, "revenue": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_restaurant_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "food_truck",
        "level": 1,
        "score": 50.0,
        "active": True,
        "history": ["started", "first_sale"],
        "stats": {"staff": 2, "revenue": 500, "customers": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_restaurant_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "food_truck",
        "level": 2,
        "score": 150.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig"],
        "stats": {"staff": 3, "revenue": 2000, "customers": 200}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_restaurant_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "pop_up",
        "level": 3,
        "score": 300.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened"],
        "stats": {"staff": 5, "revenue": 5000, "customers": 500, "rating": 4.2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_restaurant_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "pop_up",
        "level": 4,
        "score": 500.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review"],
        "stats": {"staff": 5, "revenue": 15000, "customers": 1500, "rating": 4.5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_restaurant_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "restaurant",
        "level": 5,
        "score": 1000.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened"],
        "stats": {"staff": 12, "revenue": 30000, "customers": 3000, "rating": 4.5, "seats": 40}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_restaurant_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "restaurant",
        "level": 6,
        "score": 1500.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened", "chef_hired"],
        "stats": {"staff": 15, "revenue": 50000, "customers": 5000, "rating": 4.6, "seats": 40}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_restaurant_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "restaurant",
        "level": 7,
        "score": 2500.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened", "chef_hired", "award_nominated"],
        "stats": {"staff": 15, "revenue": 80000, "customers": 8000, "rating": 4.7, "seats": 40, "awards": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_restaurant_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "acclaimed",
        "level": 8,
        "score": 4000.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened", "chef_hired", "award_nominated", "award_won"],
        "stats": {"staff": 18, "revenue": 120000, "customers": 12000, "rating": 4.8, "seats": 50, "awards": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_restaurant_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "acclaimed",
        "level": 9,
        "score": 6000.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened", "chef_hired", "award_nominated", "award_won", "expansion_planned"],
        "stats": {"staff": 18, "revenue": 180000, "customers": 18000, "rating": 4.8, "seats": 50, "awards": 1, "locations": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_restaurant_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "chain",
        "level": 10,
        "score": 8000.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened", "chef_hired", "award_nominated", "award_won", "expansion_planned", "second_location"],
        "stats": {"staff": 30, "revenue": 300000, "customers": 30000, "rating": 4.8, "seats": 90, "awards": 1, "locations": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_restaurant_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "chain",
        "level": 11,
        "score": 10000.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened", "chef_hired", "award_nominated", "award_won", "expansion_planned", "second_location", "franchise_model"],
        "stats": {"staff": 50, "revenue": 500000, "customers": 50000, "rating": 4.7, "seats": 150, "awards": 2, "locations": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_restaurant_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "chain",
        "level": 12,
        "score": 15000.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened", "chef_hired", "award_nominated", "award_won", "expansion_planned", "second_location", "franchise_model", "tv_appearance"],
        "stats": {"staff": 80, "revenue": 1000000, "customers": 100000, "rating": 4.7, "seats": 250, "awards": 2, "locations": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_restaurant_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites",
        "status": "empire",
        "level": 13,
        "score": 20000.0,
        "active": True,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened", "chef_hired", "award_nominated", "award_won", "expansion_planned", "second_location", "franchise_model", "tv_appearance", "international"],
        "stats": {"staff": 150, "revenue": 2000000, "customers": 200000, "rating": 4.6, "seats": 500, "awards": 3, "locations": 20}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_restaurant_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Tasty Bites [MICHELIN]",
        "status": "legendary",
        "level": 15,
        "score": 30000.0,
        "active": False,
        "history": ["started", "first_sale", "catering_gig", "pop_up_opened", "viral_review", "restaurant_opened", "chef_hired", "award_nominated", "award_won", "expansion_planned", "second_location", "franchise_model", "tv_appearance", "international", "michelin_star"],
        "stats": {"staff": 200, "revenue": 5000000, "customers": 500000, "rating": 4.9, "seats": 600, "awards": 5, "locations": 25, "stars": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")


# ============================================================================
# Conversation 3: Music Career (15 turns)
# ============================================================================

def validate_music_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "amateur",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["started_band"],
        "stats": {"fans": 0, "songs": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_music_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "amateur",
        "level": 1,
        "score": 10.0,
        "active": True,
        "history": ["started_band", "first_song"],
        "stats": {"fans": 10, "songs": 1, "streams": 100}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_music_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "amateur",
        "level": 2,
        "score": 30.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig"],
        "stats": {"fans": 50, "songs": 3, "streams": 500, "gigs": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_music_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "local",
        "level": 3,
        "score": 80.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following"],
        "stats": {"fans": 200, "songs": 5, "streams": 2000, "gigs": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_music_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "local",
        "level": 4,
        "score": 180.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released"],
        "stats": {"fans": 500, "songs": 8, "streams": 10000, "gigs": 10, "albums": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_music_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "signed",
        "level": 5,
        "score": 500.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal"],
        "stats": {"fans": 2000, "songs": 8, "streams": 50000, "gigs": 15, "albums": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_music_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "signed",
        "level": 6,
        "score": 1000.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal", "debut_album"],
        "stats": {"fans": 10000, "songs": 15, "streams": 200000, "gigs": 20, "albums": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_music_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "touring",
        "level": 7,
        "score": 2000.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal", "debut_album", "first_tour"],
        "stats": {"fans": 25000, "songs": 15, "streams": 500000, "gigs": 40, "albums": 2, "tours": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_music_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "touring",
        "level": 8,
        "score": 4000.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal", "debut_album", "first_tour", "hit_single"],
        "stats": {"fans": 100000, "songs": 18, "streams": 2000000, "gigs": 50, "albums": 2, "tours": 1, "hits": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_music_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "famous",
        "level": 9,
        "score": 8000.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal", "debut_album", "first_tour", "hit_single", "award_nomination"],
        "stats": {"fans": 250000, "songs": 20, "streams": 5000000, "gigs": 60, "albums": 3, "tours": 2, "hits": 2, "awards": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_music_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "famous",
        "level": 10,
        "score": 12000.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal", "debut_album", "first_tour", "hit_single", "award_nomination", "award_won"],
        "stats": {"fans": 500000, "songs": 22, "streams": 10000000, "gigs": 75, "albums": 3, "tours": 2, "hits": 3, "awards": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_music_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "star",
        "level": 11,
        "score": 20000.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal", "debut_album", "first_tour", "hit_single", "award_nomination", "award_won", "world_tour"],
        "stats": {"fans": 1000000, "songs": 25, "streams": 25000000, "gigs": 100, "albums": 4, "tours": 3, "hits": 5, "awards": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_music_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "star",
        "level": 12,
        "score": 35000.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal", "debut_album", "first_tour", "hit_single", "award_nomination", "award_won", "world_tour", "platinum_album"],
        "stats": {"fans": 2000000, "songs": 28, "streams": 50000000, "gigs": 120, "albums": 4, "tours": 3, "hits": 7, "awards": 3, "platinum": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_music_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna",
        "status": "superstar",
        "level": 14,
        "score": 50000.0,
        "active": True,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal", "debut_album", "first_tour", "hit_single", "award_nomination", "award_won", "world_tour", "platinum_album", "headliner"],
        "stats": {"fans": 5000000, "songs": 32, "streams": 100000000, "gigs": 150, "albums": 5, "tours": 5, "hits": 10, "awards": 5, "platinum": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_music_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Luna [HALL OF FAME]",
        "status": "legend",
        "level": 15,
        "score": 100000.0,
        "active": False,
        "history": ["started_band", "first_song", "first_gig", "local_following", "ep_released", "record_deal", "debut_album", "first_tour", "hit_single", "award_nomination", "award_won", "world_tour", "platinum_album", "headliner", "hall_of_fame"],
        "stats": {"fans": 10000000, "songs": 40, "streams": 200000000, "gigs": 200, "albums": 6, "tours": 6, "hits": 15, "awards": 10, "platinum": 5, "diamond": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")


# ============================================================================
# Conversation 4: Clinical Trial (15 turns)
# ============================================================================

def validate_trial_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "discovery",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["discovered"],
        "stats": {"patients": 0, "trials": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_trial_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "preclinical",
        "level": 2,
        "score": 100.0,
        "active": True,
        "history": ["discovered", "preclinical_start"],
        "stats": {"patients": 0, "trials": 0, "mice": 50, "efficacy": 60}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_trial_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "ind_filing",
        "level": 3,
        "score": 250.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success"],
        "stats": {"patients": 0, "trials": 0, "mice": 100, "efficacy": 70, "toxicity": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_trial_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "phase1",
        "level": 4,
        "score": 500.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start"],
        "stats": {"patients": 20, "trials": 1, "mice": 100, "efficacy": 70, "toxicity": 10, "sites": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_trial_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "phase1",
        "level": 5,
        "score": 1000.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete"],
        "stats": {"patients": 30, "trials": 1, "mice": 100, "efficacy": 75, "toxicity": 8, "sites": 3, "adverse_events": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_trial_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "phase2",
        "level": 6,
        "score": 1500.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start"],
        "stats": {"patients": 100, "trials": 2, "mice": 100, "efficacy": 75, "toxicity": 8, "sites": 8, "adverse_events": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_trial_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "phase2",
        "level": 7,
        "score": 2000.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start", "phase2_interim"],
        "stats": {"patients": 200, "trials": 2, "mice": 100, "efficacy": 80, "toxicity": 8, "sites": 8, "adverse_events": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_trial_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "phase2",
        "level": 8,
        "score": 4000.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start", "phase2_interim", "phase2_complete"],
        "stats": {"patients": 250, "trials": 2, "mice": 100, "efficacy": 85, "toxicity": 6, "sites": 8, "adverse_events": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_trial_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "phase3",
        "level": 9,
        "score": 6000.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start", "phase2_interim", "phase2_complete", "phase3_start"],
        "stats": {"patients": 750, "trials": 3, "mice": 100, "efficacy": 85, "toxicity": 6, "sites": 28, "adverse_events": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_trial_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "phase3",
        "level": 10,
        "score": 8000.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start", "phase2_interim", "phase2_complete", "phase3_start", "phase3_enrolled"],
        "stats": {"patients": 2250, "trials": 3, "mice": 100, "efficacy": 85, "toxicity": 6, "sites": 58, "adverse_events": 5, "countries": 12}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_trial_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "phase3",
        "level": 11,
        "score": 10000.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start", "phase2_interim", "phase2_complete", "phase3_start", "phase3_enrolled", "phase3_interim"],
        "stats": {"patients": 2750, "trials": 3, "mice": 100, "efficacy": 88, "toxicity": 6, "sites": 58, "adverse_events": 8, "countries": 12}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_trial_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "nda_prep",
        "level": 12,
        "score": 20000.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start", "phase2_interim", "phase2_complete", "phase3_start", "phase3_enrolled", "phase3_interim", "phase3_success"],
        "stats": {"patients": 3250, "trials": 3, "mice": 100, "efficacy": 92, "toxicity": 5, "sites": 58, "adverse_events": 8, "countries": 12}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_trial_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "fda_review",
        "level": 13,
        "score": 21000.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start", "phase2_interim", "phase2_complete", "phase3_start", "phase3_enrolled", "phase3_interim", "phase3_success", "nda_submitted"],
        "stats": {"patients": 3250, "trials": 3, "mice": 100, "efficacy": 92, "toxicity": 5, "sites": 58, "adverse_events": 8, "countries": 12, "reviewers": 15}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_trial_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Nexavir",
        "status": "fda_review",
        "level": 14,
        "score": 22000.0,
        "active": True,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start", "phase2_interim", "phase2_complete", "phase3_start", "phase3_enrolled", "phase3_interim", "phase3_success", "nda_submitted", "advisory_positive"],
        "stats": {"patients": 3250, "trials": 3, "mice": 100, "efficacy": 92, "toxicity": 5, "sites": 58, "adverse_events": 8, "countries": 12, "reviewers": 15, "votes": 12}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_trial_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "[FDA] Nexavir",
        "status": "approved",
        "level": 15,
        "score": 66000.0,
        "active": False,
        "history": ["discovered", "preclinical_start", "preclinical_success", "phase1_start", "phase1_complete", "phase2_start", "phase2_interim", "phase2_complete", "phase3_start", "phase3_enrolled", "phase3_interim", "phase3_success", "nda_submitted", "advisory_positive", "fda_approved"],
        "stats": {"patients": 3250, "trials": 3, "mice": 100, "efficacy": 92, "toxicity": 5, "sites": 58, "adverse_events": 8, "countries": 12, "reviewers": 15, "votes": 12, "approval_year": 2024}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")


# ============================================================================
# Conversation 5: Game Development (15 turns)
# ============================================================================

def validate_game_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "concept",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["conceived"],
        "stats": {"devs": 3, "bugs": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_game_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "prototype",
        "level": 2,
        "score": 50.0,
        "active": True,
        "history": ["conceived", "prototype_done"],
        "stats": {"devs": 5, "bugs": 20, "features": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_game_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "vertical_slice",
        "level": 3,
        "score": 150.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice"],
        "stats": {"devs": 8, "bugs": 50, "features": 15, "playtime": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_game_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "alpha",
        "level": 4,
        "score": 300.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build"],
        "stats": {"devs": 12, "bugs": 100, "features": 30, "playtime": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_game_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "alpha",
        "level": 5,
        "score": 500.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded"],
        "stats": {"devs": 20, "bugs": 150, "features": 40, "playtime": 5, "artists": 8}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")

def validate_game_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "beta",
        "level": 6,
        "score": 1000.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build"],
        "stats": {"devs": 25, "bugs": 250, "features": 60, "playtime": 10, "artists": 12}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 6 complete")

def validate_game_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "beta",
        "level": 7,
        "score": 1500.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build", "closed_beta"],
        "stats": {"devs": 25, "bugs": 200, "features": 65, "playtime": 12, "artists": 12, "testers": 500}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 7 complete")

def validate_game_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "beta",
        "level": 8,
        "score": 2000.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build", "closed_beta", "open_beta"],
        "stats": {"devs": 25, "bugs": 500, "features": 70, "playtime": 15, "artists": 12, "testers": 10000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 8 complete")

def validate_game_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "beta",
        "level": 9,
        "score": 2500.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build", "closed_beta", "open_beta", "bug_fixing"],
        "stats": {"devs": 30, "bugs": 100, "features": 72, "playtime": 15, "artists": 12, "testers": 15000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 9 complete")

def validate_game_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "gold",
        "level": 10,
        "score": 5000.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build", "closed_beta", "open_beta", "bug_fixing", "gold_master"],
        "stats": {"devs": 30, "bugs": 30, "features": 72, "playtime": 20, "artists": 12, "testers": 20000}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 10 complete")

def validate_game_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "gold",
        "level": 11,
        "score": 5500.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build", "closed_beta", "open_beta", "bug_fixing", "gold_master", "day_one_patch"],
        "stats": {"devs": 30, "bugs": 10, "features": 75, "playtime": 20, "artists": 12, "testers": 20000, "patches": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 11 complete")

def validate_game_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "launched",
        "level": 12,
        "score": 11000.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build", "closed_beta", "open_beta", "bug_fixing", "gold_master", "day_one_patch", "launched"],
        "stats": {"devs": 30, "bugs": 30, "features": 75, "playtime": 20, "artists": 12, "testers": 20000, "patches": 1, "sales": 100000, "reviews": 85}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 12 complete")

def validate_game_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "launched",
        "level": 13,
        "score": 14000.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build", "closed_beta", "open_beta", "bug_fixing", "gold_master", "day_one_patch", "launched", "post_launch"],
        "stats": {"devs": 30, "bugs": 15, "features": 80, "playtime": 20, "artists": 12, "testers": 20000, "patches": 3, "sales": 500000, "reviews": 88}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 13 complete")

def validate_game_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal",
        "status": "launched",
        "level": 14,
        "score": 17000.0,
        "active": True,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build", "closed_beta", "open_beta", "bug_fixing", "gold_master", "day_one_patch", "launched", "post_launch", "dlc_released"],
        "stats": {"devs": 30, "bugs": 20, "features": 100, "playtime": 30, "artists": 12, "testers": 20000, "patches": 4, "sales": 800000, "reviews": 90, "dlc": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 14 complete")

def validate_game_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Ethereal [GOTY]",
        "status": "legendary",
        "level": 15,
        "score": 51000.0,
        "active": False,
        "history": ["conceived", "prototype_done", "vertical_slice", "alpha_build", "team_expanded", "beta_build", "closed_beta", "open_beta", "bug_fixing", "gold_master", "day_one_patch", "launched", "post_launch", "dlc_released", "goty_winner"],
        "stats": {"devs": 30, "bugs": 20, "features": 100, "playtime": 30, "artists": 12, "testers": 20000, "patches": 4, "sales": 1600000, "reviews": 95, "dlc": 2, "awards": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 15 complete")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("name", name, "Entity name/identifier (str, initial: ''). Use exact quoted name from query."),
    Variable("status", status, "Current status/phase as a single lowercase word or snake_case (str, initial: ''). Examples: 'trainee', 'cadet', 'commander', 'food_truck', 'pop_up', 'restaurant', 'amateur', 'local', 'signed', 'discovery', 'phase1', 'concept', 'prototype', 'alpha', 'beta', 'gold', 'launched'. Extract the key status word, not the full phrase."),
    Variable("level", level, "Current level (int, initial: 0)."),
    Variable("score", score, "Cumulative score/progress (float, initial: 0.0)."),
    Variable("active", active, "Whether entity is still active (bool, initial: False). Set to False when complete."),
    Variable("history", history, "List of event identifiers (list[str], initial: []). Append one snake_case event per turn. Examples: 'accepted', 'training_start', 'grand_opening', 'first_song', 'discovered', 'conceived'."),
    Variable("stats", stats, """Additional statistics (dict, initial: {}). Only use these exact keys based on conversation:
- space_exploration: crew, fuel, altitude, samples, moonwalks, base, medals
- restaurant_business: staff, revenue, customers, rating, seats, awards, locations, stars
- music_career: fans, songs, streams, gigs, albums, tours, hits, awards, platinum, diamond
- clinical_trial: patients, trials, mice, efficacy, toxicity, sites, adverse_events, countries, reviewers, votes, approval_year
- game_development: devs, bugs, features, playtime, artists, testers, patches, sales, reviews, dlc, awards
Values are numbers (int/float), strings, or booleans. Do NOT add keys not in this list."""),
]

validators = {
    # Space exploration
    "validate_space_t1": validate_space_t1,
    "validate_space_t2": validate_space_t2,
    "validate_space_t3": validate_space_t3,
    "validate_space_t4": validate_space_t4,
    "validate_space_t5": validate_space_t5,
    "validate_space_t6": validate_space_t6,
    "validate_space_t7": validate_space_t7,
    "validate_space_t8": validate_space_t8,
    "validate_space_t9": validate_space_t9,
    "validate_space_t10": validate_space_t10,
    "validate_space_t11": validate_space_t11,
    "validate_space_t12": validate_space_t12,
    "validate_space_t13": validate_space_t13,
    "validate_space_t14": validate_space_t14,
    "validate_space_t15": validate_space_t15,
    # Restaurant business
    "validate_restaurant_t1": validate_restaurant_t1,
    "validate_restaurant_t2": validate_restaurant_t2,
    "validate_restaurant_t3": validate_restaurant_t3,
    "validate_restaurant_t4": validate_restaurant_t4,
    "validate_restaurant_t5": validate_restaurant_t5,
    "validate_restaurant_t6": validate_restaurant_t6,
    "validate_restaurant_t7": validate_restaurant_t7,
    "validate_restaurant_t8": validate_restaurant_t8,
    "validate_restaurant_t9": validate_restaurant_t9,
    "validate_restaurant_t10": validate_restaurant_t10,
    "validate_restaurant_t11": validate_restaurant_t11,
    "validate_restaurant_t12": validate_restaurant_t12,
    "validate_restaurant_t13": validate_restaurant_t13,
    "validate_restaurant_t14": validate_restaurant_t14,
    "validate_restaurant_t15": validate_restaurant_t15,
    # Music career
    "validate_music_t1": validate_music_t1,
    "validate_music_t2": validate_music_t2,
    "validate_music_t3": validate_music_t3,
    "validate_music_t4": validate_music_t4,
    "validate_music_t5": validate_music_t5,
    "validate_music_t6": validate_music_t6,
    "validate_music_t7": validate_music_t7,
    "validate_music_t8": validate_music_t8,
    "validate_music_t9": validate_music_t9,
    "validate_music_t10": validate_music_t10,
    "validate_music_t11": validate_music_t11,
    "validate_music_t12": validate_music_t12,
    "validate_music_t13": validate_music_t13,
    "validate_music_t14": validate_music_t14,
    "validate_music_t15": validate_music_t15,
    # Clinical trial
    "validate_trial_t1": validate_trial_t1,
    "validate_trial_t2": validate_trial_t2,
    "validate_trial_t3": validate_trial_t3,
    "validate_trial_t4": validate_trial_t4,
    "validate_trial_t5": validate_trial_t5,
    "validate_trial_t6": validate_trial_t6,
    "validate_trial_t7": validate_trial_t7,
    "validate_trial_t8": validate_trial_t8,
    "validate_trial_t9": validate_trial_t9,
    "validate_trial_t10": validate_trial_t10,
    "validate_trial_t11": validate_trial_t11,
    "validate_trial_t12": validate_trial_t12,
    "validate_trial_t13": validate_trial_t13,
    "validate_trial_t14": validate_trial_t14,
    "validate_trial_t15": validate_trial_t15,
    # Game development
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
    "validate_game_t11": validate_game_t11,
    "validate_game_t12": validate_game_t12,
    "validate_game_t13": validate_game_t13,
    "validate_game_t14": validate_game_t14,
    "validate_game_t15": validate_game_t15,
}
