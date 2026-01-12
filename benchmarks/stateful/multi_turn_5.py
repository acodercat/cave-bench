"""
Multi-Turn (5) - Stateful Benchmark

Tests: Managing state across 5 turns with consistent variable updates.
Focus is on state persistence and correct modifications over multiple interactions.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# INITIAL STATE - 6 variables for tracking state across turns
# ============================================================================

# Strings
name = ""
status = ""

# Integers
level = 0

# Floats
score = 0.0

# Booleans
active = False

# Collections
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
# Conversation 1: Player Progression
# ============================================================================

def validate_player_turn1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "beginner",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["joined"],
        "stats": {"health": 100, "mana": 50}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_player_turn2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Alex",
        "status": "adventurer",
        "level": 2,
        "score": 100.0,
        "active": True,
        "history": ["joined", "first_quest"],
        "stats": {"health": 100, "mana": 50, "xp": 100}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_player_turn3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "[GUILD] Alex",
        "status": "adventurer",
        "level": 3,
        "score": 250.0,
        "active": True,
        "history": ["joined", "first_quest", "joined_guild"],
        "stats": {"health": 100, "mana": 50, "xp": 250, "guild": "Phoenix"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_player_turn4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "[GUILD] Alex",
        "status": "champion",
        "level": 5,
        "score": 750.0,
        "active": True,
        "history": ["joined", "first_quest", "joined_guild", "tournament_win"],
        "stats": {"health": 100, "mana": 50, "xp": 750, "guild": "Phoenix", "trophies": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_player_turn5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "[LEGEND] Alex",
        "status": "legend",
        "level": 10,
        "score": 1500.0,
        "active": True,
        "history": ["joined", "first_quest", "joined_guild", "tournament_win", "legendary"],
        "stats": {"health": 100, "mana": 50, "xp": 1500, "guild": "Phoenix", "trophies": 1, "legend_rank": 1}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")


# ============================================================================
# Conversation 2: Project Lifecycle
# ============================================================================

def validate_project_turn1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Website Redesign",
        "status": "planning",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["created"],
        "stats": {"tasks": 0, "completed": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_project_turn2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Website Redesign",
        "status": "in_progress",
        "level": 2,
        "score": 20.0,
        "active": True,
        "history": ["created", "started"],
        "stats": {"tasks": 10, "completed": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_project_turn3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Website Redesign",
        "status": "review",
        "level": 3,
        "score": 60.0,
        "active": True,
        "history": ["created", "started", "review_requested"],
        "stats": {"tasks": 10, "completed": 6, "reviewers": 2}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_project_turn4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Website Redesign",
        "status": "testing",
        "level": 4,
        "score": 85.0,
        "active": True,
        "history": ["created", "started", "review_requested", "approved"],
        "stats": {"tasks": 10, "completed": 9, "reviewers": 2, "bugs": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_project_turn5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Website Redesign v1.0",
        "status": "complete",
        "level": 5,
        "score": 100.0,
        "active": False,
        "history": ["created", "started", "review_requested", "approved", "deployed"],
        "stats": {"tasks": 10, "completed": 10, "reviewers": 2, "bugs": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")


# ============================================================================
# Conversation 3: Order Fulfillment
# ============================================================================

def validate_order_turn1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ORD-12345",
        "status": "pending",
        "level": 1,
        "score": 149.99,
        "active": True,
        "history": ["placed"],
        "stats": {"items": 3, "weight": 2.5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_order_turn2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ORD-12345",
        "status": "confirmed",
        "level": 2,
        "score": 149.99,
        "active": True,
        "history": ["placed", "payment_confirmed"],
        "stats": {"items": 3, "weight": 2.5, "payment_id": "PAY-789"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_order_turn3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ORD-12345",
        "status": "shipped",
        "level": 3,
        "score": 149.99,
        "active": True,
        "history": ["placed", "payment_confirmed", "shipped"],
        "stats": {"items": 3, "weight": 2.5, "payment_id": "PAY-789", "tracking": "TRK-456"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_order_turn4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ORD-12345",
        "status": "out_for_delivery",
        "level": 4,
        "score": 149.99,
        "active": True,
        "history": ["placed", "payment_confirmed", "shipped", "out_for_delivery"],
        "stats": {"items": 3, "weight": 2.5, "payment_id": "PAY-789", "tracking": "TRK-456", "driver": "John"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_order_turn5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "ORD-12345",
        "status": "delivered",
        "level": 5,
        "score": 149.99,
        "active": False,
        "history": ["placed", "payment_confirmed", "shipped", "out_for_delivery", "delivered"],
        "stats": {"items": 3, "weight": 2.5, "payment_id": "PAY-789", "tracking": "TRK-456", "driver": "John", "rating": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")


# ============================================================================
# Conversation 4: Fitness Journey
# ============================================================================

def validate_fitness_turn1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Sam",
        "status": "beginner",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["signed_up"],
        "stats": {"workouts": 0, "calories": 0}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_fitness_turn2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Sam",
        "status": "beginner",
        "level": 1,
        "score": 50.0,
        "active": True,
        "history": ["signed_up", "first_workout"],
        "stats": {"workouts": 1, "calories": 300}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_fitness_turn3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Sam",
        "status": "intermediate",
        "level": 2,
        "score": 200.0,
        "active": True,
        "history": ["signed_up", "first_workout", "week_streak"],
        "stats": {"workouts": 7, "calories": 2100, "streak": 7}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_fitness_turn4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Sam",
        "status": "advanced",
        "level": 3,
        "score": 500.0,
        "active": True,
        "history": ["signed_up", "first_workout", "week_streak", "month_milestone"],
        "stats": {"workouts": 30, "calories": 9000, "streak": 30, "badges": 3}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_fitness_turn5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "Sam [ELITE]",
        "status": "elite",
        "level": 5,
        "score": 1000.0,
        "active": True,
        "history": ["signed_up", "first_workout", "week_streak", "month_milestone", "elite_status"],
        "stats": {"workouts": 100, "calories": 30000, "streak": 100, "badges": 10}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")


# ============================================================================
# Conversation 5: Support Ticket
# ============================================================================

def validate_ticket_turn1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TKT-9999",
        "status": "open",
        "level": 1,
        "score": 0.0,
        "active": True,
        "history": ["created"],
        "stats": {"priority": 3, "category": "billing"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 1 complete")

def validate_ticket_turn2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TKT-9999",
        "status": "assigned",
        "level": 2,
        "score": 0.0,
        "active": True,
        "history": ["created", "assigned"],
        "stats": {"priority": 3, "category": "billing", "agent": "Lisa"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 2 complete")

def validate_ticket_turn3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TKT-9999 [ESCALATED]",
        "status": "escalated",
        "level": 3,
        "score": 0.0,
        "active": True,
        "history": ["created", "assigned", "escalated"],
        "stats": {"priority": 1, "category": "billing", "agent": "Lisa", "escalated_to": "Manager"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 3 complete")

def validate_ticket_turn4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TKT-9999 [ESCALATED]",
        "status": "in_progress",
        "level": 4,
        "score": 50.0,
        "active": True,
        "history": ["created", "assigned", "escalated", "solution_found"],
        "stats": {"priority": 1, "category": "billing", "agent": "Lisa", "escalated_to": "Manager", "resolution": "refund"}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 4 complete")

def validate_ticket_turn5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "name": "TKT-9999 [RESOLVED]",
        "status": "closed",
        "level": 5,
        "score": 100.0,
        "active": False,
        "history": ["created", "assigned", "escalated", "solution_found", "resolved"],
        "stats": {"priority": 1, "category": "billing", "agent": "Lisa", "escalated_to": "Manager", "resolution": "refund", "satisfaction": 5}
    }
    errors = _validate_state(runtime, expected)
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Turn 5 complete")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("name", name, "Entity name/identifier (str, initial: '')."),
    Variable("status", status, "Current status (str, initial: ''). Valid values depend on context (e.g., 'beginner', 'intermediate', 'advanced', 'elite' for progression; 'pending', 'confirmed', 'shipped', 'delivered' for orders; 'open', 'assigned', 'escalated', 'closed' for tickets)."),
    Variable("level", level, "Current level/phase (int, initial: 0). Typically increments with each turn."),
    Variable("score", score, "Cumulative score/progress (float, initial: 0.0)."),
    Variable("active", active, "Whether the entity is still active (bool, initial: False). Set to False when complete/closed."),
    Variable("history", history, "List of events/actions taken (list[str], initial: []). Append one event per turn."),
    Variable("stats", stats, """Additional statistics (dict, initial: {}). Only use these exact keys based on conversation:
- player_progression: health, mana, xp, guild, trophies, legend_rank
- project_lifecycle: tasks, completed, reviewers, bugs
- order_fulfillment: items, weight, payment_id, tracking, driver, rating
- fitness_journey: workouts, calories, streak, badges
- support_ticket: priority, category, agent, escalated_to, resolution, satisfaction
Values are numbers (int/float) or strings. Do NOT add keys not in this list."""),
]

validators = {
    # Player progression
    "validate_player_turn1": validate_player_turn1,
    "validate_player_turn2": validate_player_turn2,
    "validate_player_turn3": validate_player_turn3,
    "validate_player_turn4": validate_player_turn4,
    "validate_player_turn5": validate_player_turn5,
    # Project lifecycle
    "validate_project_turn1": validate_project_turn1,
    "validate_project_turn2": validate_project_turn2,
    "validate_project_turn3": validate_project_turn3,
    "validate_project_turn4": validate_project_turn4,
    "validate_project_turn5": validate_project_turn5,
    # Order fulfillment
    "validate_order_turn1": validate_order_turn1,
    "validate_order_turn2": validate_order_turn2,
    "validate_order_turn3": validate_order_turn3,
    "validate_order_turn4": validate_order_turn4,
    "validate_order_turn5": validate_order_turn5,
    # Fitness journey
    "validate_fitness_turn1": validate_fitness_turn1,
    "validate_fitness_turn2": validate_fitness_turn2,
    "validate_fitness_turn3": validate_fitness_turn3,
    "validate_fitness_turn4": validate_fitness_turn4,
    "validate_fitness_turn5": validate_fitness_turn5,
    # Support ticket
    "validate_ticket_turn1": validate_ticket_turn1,
    "validate_ticket_turn2": validate_ticket_turn2,
    "validate_ticket_turn3": validate_ticket_turn3,
    "validate_ticket_turn4": validate_ticket_turn4,
    "validate_ticket_turn5": validate_ticket_turn5,
}
