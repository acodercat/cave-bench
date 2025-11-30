"""
Multi-Variable (10) - Stateful Benchmark

Tests: Managing 10 variables of diverse types modified in each turn.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# INITIAL STATE - 10 variables with diverse types
# ============================================================================

# Strings
product_name = ""
category = ""

# Integers
quantity = 0
stock_count = 0

# Floats
price = 0.0
discount_rate = 0.0

# Booleans
in_stock = False
is_featured = False

# Collections
reviews = []
metadata = {}


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_product_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "product_name": "Laptop",
        "category": "Electronics",
        "quantity": 50,
        "stock_count": 100,
        "price": 999.99,
        "discount_rate": 0.1,
        "in_stock": True,
        "is_featured": True,
        "reviews": [5, 4, 5],
        "metadata": {"brand": "TechCo", "year": 2024}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Product initialized")

def validate_product_sale(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "product_name": "Laptop",
        "category": "Electronics",
        "quantity": 45,
        "stock_count": 95,
        "price": 899.99,
        "discount_rate": 0.1,
        "in_stock": True,
        "is_featured": True,
        "reviews": [5, 4, 5, 4],
        "metadata": {"brand": "TechCo", "year": 2024, "sold": 5}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Sale processed")

def validate_product_update(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "product_name": "Laptop Pro",
        "category": "Premium Electronics",
        "quantity": 45,
        "stock_count": 95,
        "price": 1079.99,
        "discount_rate": 0.2,
        "in_stock": True,
        "is_featured": False,
        "reviews": [5, 4, 5, 4],
        "metadata": {"brand": "TechCo", "year": 2024, "sold": 5, "upgraded": True}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Product updated")


def validate_inventory_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "product_name": "Phone",
        "category": "Mobile",
        "quantity": 200,
        "stock_count": 500,
        "price": 599.0,
        "discount_rate": 0.0,
        "in_stock": True,
        "is_featured": False,
        "reviews": [],
        "metadata": {"color": "black"}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Inventory initialized")

def validate_inventory_restock(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "product_name": "Phone",
        "category": "Mobile",
        "quantity": 400,
        "stock_count": 700,
        "price": 599.0,
        "discount_rate": 0.0,
        "in_stock": True,
        "is_featured": True,
        "reviews": [5],
        "metadata": {"color": "black", "restocked": True}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Restocked")

def validate_inventory_clearance(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "product_name": "Phone [CLEARANCE]",
        "category": "Mobile",
        "quantity": 0,
        "stock_count": 0,
        "price": 299.5,
        "discount_rate": 0.5,
        "in_stock": False,
        "is_featured": False,
        "reviews": [5],
        "metadata": {"color": "black", "restocked": True, "clearance": True}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Clearance applied")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("product_name", product_name, "Product name (str, initial: '')."),
    Variable("category", category, "Product category (str, initial: '')."),
    Variable("quantity", quantity, "Available quantity (int, initial: 0)."),
    Variable("stock_count", stock_count, "Total stock count (int, initial: 0)."),
    Variable("price", price, "Product price (float, initial: 0.0)."),
    Variable("discount_rate", discount_rate, "Discount rate 0-1 (float, initial: 0.0)."),
    Variable("in_stock", in_stock, "In stock status (bool, initial: False)."),
    Variable("is_featured", is_featured, "Featured status (bool, initial: False)."),
    Variable("reviews", reviews, "Review scores (list, initial: [])."),
    Variable("metadata", metadata, "Product metadata (dict, initial: {})."),
]

validators = {
    "validate_product_init": validate_product_init,
    "validate_product_sale": validate_product_sale,
    "validate_product_update": validate_product_update,
    "validate_inventory_init": validate_inventory_init,
    "validate_inventory_restock": validate_inventory_restock,
    "validate_inventory_clearance": validate_inventory_clearance,
}
