"""
Multi-Variable (20) - Stateful Benchmark

Tests: Managing 20 variables of diverse types modified in each turn.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# INITIAL STATE - 20 variables with diverse types
# ============================================================================

# Strings (4)
company_name = ""
industry = ""
ceo = ""
headquarters = ""

# Integers (4)
employees = 0
founded_year = 0
offices = 0
products = 0

# Floats (4)
revenue = 0.0
profit_margin = 0.0
stock_price = 0.0
market_cap = 0.0

# Booleans (4)
public = False
profitable = False
hiring = False
international = False

# Collections (4)
departments = []
locations = []
financials = {}
contacts = {}


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_startup_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "company_name": "TechStart",
        "industry": "Software",
        "ceo": "Alice Johnson",
        "headquarters": "San Francisco",
        "employees": 50,
        "founded_year": 2020,
        "offices": 1,
        "products": 2,
        "revenue": 5000000.0,
        "profit_margin": 0.1,
        "stock_price": 0.0,
        "market_cap": 0.0,
        "public": False,
        "profitable": True,
        "hiring": True,
        "international": False,
        "departments": ["Engineering", "Sales", "Marketing"],
        "locations": ["SF"],
        "financials": {"funding": 10000000, "round": "Series A"},
        "contacts": {"email": "info@techstart.com", "phone": "555-0100"}
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
    return ValidatorResult(True, "Startup initialized")

def validate_startup_growth(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "company_name": "TechStart",
        "industry": "Software",
        "ceo": "Alice Johnson",
        "headquarters": "San Francisco",
        "employees": 150,
        "founded_year": 2020,
        "offices": 3,
        "products": 5,
        "revenue": 15000000.0,
        "profit_margin": 0.15,
        "stock_price": 0.0,
        "market_cap": 0.0,
        "public": False,
        "profitable": True,
        "hiring": True,
        "international": True,
        "departments": ["Engineering", "Sales", "Marketing", "HR", "Finance"],
        "locations": ["SF", "NYC", "London"],
        "financials": {"funding": 10000000, "round": "Series A", "valuation": 100000000},
        "contacts": {"email": "info@techstart.com", "phone": "555-0100", "support": "555-0200"}
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
    return ValidatorResult(True, "Growth phase complete")

def validate_startup_ipo(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "company_name": "TechStart Inc.",
        "industry": "Enterprise Software",
        "ceo": "Alice Johnson",
        "headquarters": "San Francisco",
        "employees": 500,
        "founded_year": 2020,
        "offices": 10,
        "products": 10,
        "revenue": 50000000.0,
        "profit_margin": 0.2,
        "stock_price": 25.0,
        "market_cap": 500000000.0,
        "public": True,
        "profitable": True,
        "hiring": True,
        "international": True,
        "departments": ["Engineering", "Sales", "Marketing", "HR", "Finance", "Legal", "IR"],
        "locations": ["SF", "NYC", "London", "Tokyo", "Berlin"],
        "financials": {"funding": 10000000, "round": "Series A", "valuation": 100000000, "ipo": True},
        "contacts": {"email": "info@techstart.com", "phone": "555-0100", "support": "555-0200", "ir": "ir@techstart.com"}
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
    return ValidatorResult(True, "IPO complete")


def validate_enterprise_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "company_name": "MegaCorp",
        "industry": "Conglomerate",
        "ceo": "Bob Williams",
        "headquarters": "New York",
        "employees": 10000,
        "founded_year": 1990,
        "offices": 50,
        "products": 100,
        "revenue": 1000000000.0,
        "profit_margin": 0.12,
        "stock_price": 150.0,
        "market_cap": 50000000000.0,
        "public": True,
        "profitable": True,
        "hiring": False,
        "international": True,
        "departments": ["All"],
        "locations": ["Global"],
        "financials": {"debt": 500000000, "assets": 2000000000},
        "contacts": {"ir": "ir@megacorp.com"}
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
    return ValidatorResult(True, "Enterprise initialized")

def validate_enterprise_restructure(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "company_name": "MegaCorp",
        "industry": "Technology",
        "ceo": "Carol Lee",
        "headquarters": "New York",
        "employees": 8000,
        "founded_year": 1990,
        "offices": 30,
        "products": 50,
        "revenue": 800000000.0,
        "profit_margin": 0.18,
        "stock_price": 180.0,
        "market_cap": 60000000000.0,
        "public": True,
        "profitable": True,
        "hiring": True,
        "international": True,
        "departments": ["Tech", "Cloud", "AI"],
        "locations": ["Americas", "Europe", "Asia"],
        "financials": {"debt": 200000000, "assets": 2500000000},
        "contacts": {"ir": "ir@megacorp.com", "press": "press@megacorp.com"}
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
    return ValidatorResult(True, "Restructure complete")

def validate_enterprise_acquire(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "company_name": "MegaCorp Global",
        "industry": "Technology",
        "ceo": "Carol Lee",
        "headquarters": "New York",
        "employees": 12000,
        "founded_year": 1990,
        "offices": 45,
        "products": 75,
        "revenue": 1200000000.0,
        "profit_margin": 0.15,
        "stock_price": 200.0,
        "market_cap": 80000000000.0,
        "public": True,
        "profitable": True,
        "hiring": True,
        "international": True,
        "departments": ["Tech", "Cloud", "AI", "Acquired"],
        "locations": ["Americas", "Europe", "Asia", "Australia"],
        "financials": {"debt": 200000000, "assets": 2500000000, "acquisition": 500000000},
        "contacts": {"ir": "ir@megacorp.com", "press": "press@megacorp.com", "ma": "ma@megacorp.com"}
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
    return ValidatorResult(True, "Acquisition complete")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("company_name", company_name, "Company name (str, initial: '')."),
    Variable("industry", industry, "Industry sector (str, initial: '')."),
    Variable("ceo", ceo, "CEO name (str, initial: '')."),
    Variable("headquarters", headquarters, "HQ city (str, initial: '')."),
    Variable("employees", employees, "Employee count (int, initial: 0)."),
    Variable("founded_year", founded_year, "Year founded (int, initial: 0)."),
    Variable("offices", offices, "Number of offices (int, initial: 0)."),
    Variable("products", products, "Product count (int, initial: 0)."),
    Variable("revenue", revenue, "Annual revenue (float, initial: 0.0)."),
    Variable("profit_margin", profit_margin, "Profit margin 0-1 (float, initial: 0.0)."),
    Variable("stock_price", stock_price, "Stock price (float, initial: 0.0)."),
    Variable("market_cap", market_cap, "Market capitalization (float, initial: 0.0)."),
    Variable("public", public, "Publicly traded (bool, initial: False)."),
    Variable("profitable", profitable, "Profitable status (bool, initial: False)."),
    Variable("hiring", hiring, "Currently hiring (bool, initial: False)."),
    Variable("international", international, "International presence (bool, initial: False)."),
    Variable("departments", departments, "Department names (list, initial: [])."),
    Variable("locations", locations, "Office locations (list, initial: [])."),
    Variable("financials", financials, "Financial data (dict, initial: {})."),
    Variable("contacts", contacts, "Contact info (dict, initial: {})."),
]

validators = {
    "validate_startup_init": validate_startup_init,
    "validate_startup_growth": validate_startup_growth,
    "validate_startup_ipo": validate_startup_ipo,
    "validate_enterprise_init": validate_enterprise_init,
    "validate_enterprise_restructure": validate_enterprise_restructure,
    "validate_enterprise_acquire": validate_enterprise_acquire,
}
