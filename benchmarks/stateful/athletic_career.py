"""
Athletic Career - Multi-Turn Stateful Benchmark

Tests: Managing an athletic career from amateur to legendary status across 20 turns.
Focus is on tracking performance metrics, sponsorships, and career milestones.
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


def validate_t1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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
    Variable("name", name, "Athlete name/identifier (str, initial: ''). Use exact quoted name from query."),
    Variable("status", status, "Current career phase as a single lowercase word (str, initial: ''). Examples: 'amateur', 'semi_pro', 'professional', 'elite', 'world_class', 'olympian', 'legend'. Extract the key status word, not the full phrase."),
    Variable("level", level, "Current level (int, initial: 0)."),
    Variable("score", score, "Cumulative score/progress (float, initial: 0.0)."),
    Variable("active", active, "Whether athlete is still competing (bool, initial: False). Set to False when retired."),
    Variable("history", history, "List of event identifiers (list[str], initial: []). Append one snake_case event per turn. Examples: 'started_training', 'first_competition', 'first_medal'."),
    Variable("stats", stats, """A flat dictionary for tracking athletic statistics (initial: {}). Set keys directly on stats like stats["events"] = 5. Valid keys:
events, medals, pb (personal best time), training_hours, sponsors, earnings, ranking, olympics, gold, hall_of_fame
Values are numbers (int/float). Do NOT create nested dicts or add keys not in this list."""),
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
