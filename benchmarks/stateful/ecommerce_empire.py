"""
E-commerce Empire - Multi-Turn Stateful Benchmark

Tests: Managing e-commerce startup progression across 20 turns.
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

def validate_t2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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

def validate_t20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
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
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("name", name, "Entity name/identifier (str, initial: ''). Use exact quoted name from query."),
    Variable("status", status, "Current status/phase as a single lowercase word (str, initial: ''). Examples: 'idea', 'launched', 'growing', 'scaling', 'funded', 'national', 'enterprise', 'pre_ipo', 'public', 'leader'."),
    Variable("level", level, "Current level (int, initial: 0)."),
    Variable("score", score, "Cumulative score/progress (float, initial: 0.0)."),
    Variable("active", active, "Whether entity is still active (bool, initial: False). Set to False when complete."),
    Variable("history", history, "List of event identifiers (list[str], initial: []). Append one snake_case event per turn."),
    Variable("stats", stats, """A flat dictionary for tracking statistics (initial: {}). Valid keys: products, orders, visitors, revenue, employees, warehouses, satisfaction, downloads, funding, countries, members, sellers, valuation, ticker. Values are numbers (int/float) or strings. Do NOT create nested dicts or add keys not in this list."""),
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
