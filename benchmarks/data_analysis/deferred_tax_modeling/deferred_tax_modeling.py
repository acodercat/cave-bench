"""
This scenario involves preparing a forecast of asset schedules for a new business
and analyzing the impact on the forecast deferred tax.

BACKGROUND
- Forecast Capital Expenditure (Capex): Provided in real 2012 US$m. All capex
  is incurred on the first day of each calendar year.
- Inflation: Forecast to be 3% per annum.
- Accounting Treatment: Depreciation is calculated on a straight-line basis
  over 12 years.
- Tax Treatment: Depreciation is calculated using the diminishing value method
  at a rate of 40% per annum.
- Tax Rate: 30% per annum, moving down to 28% from the start of 2016.

QUESTIONS & ANSWERS
1.  What is the total nominal capex for 2012 through 2021 (inclusive)?
    [Answer: b. $598.3m]
2.  What is the closing balance for the accounting asset schedule in 2017?
    [Answer: a. $298.3m]
3.  What is the closing balance for the taxation asset schedule 2019?
    [Answer: d. $68.2m]
4.  What impact does the change in taxation rate have on the opening deferred
    tax balance for 2016?
    [Answer: a. $2.2m]
5.  What is the deferred tax balance as at 2028?
    [Answer: b. $79.0m Liability]
"""

from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn
from typing import List
from core.types import ToolCall

# Capex in real 2012 $m from 2012 to 2029
real_capex_schedule = {
    2012: 10, 2013: 64, 2014: 69, 2015: 99, 2016: 89, 2017: 39, 2018: 34,
    2019: 23, 2020: 29, 2021: 69, 2022: 99, 2023: 89, 2024: 39, 2025: 34,
    2026: 23, 2027: 29, 2028: 69, 2029: 99
}

# General Assumptions
inflation_rate = 0.03
accounting_depreciation_life = 12  # years
tax_depreciation_rate = 0.40
tax_rate_pre_2016 = 0.30
tax_rate_post_2016 = 0.28


tools = []


# Correct answers are based on calculations from the problem description.
CORRECT_ANSWERS = {
    "q1": 598.3,
    "q2": 298.3,
    "q3": 68.2,
    "q4": 2.2,
    "q5": 79.0,
    "q5_type": "Liability"
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Validator for total nominal capex 2012-2021."""
    value = runtime.retrieve("q1_total_nominal_capex")
    expected = CORRECT_ANSWERS["q1"]
    if value == expected:
        return ValidatorResult(success=True, message="Calculation for total nominal capex is correct.")
    else:
        return ValidatorResult(success=False, message=f"Incorrect. Expected a value close to {expected}, but got {value}.")

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Validator for accounting closing balance in 2017."""
    value = runtime.retrieve("q2_accounting_closing_balance_2017")
    expected = CORRECT_ANSWERS["q2"]
    if value == expected:
        return ValidatorResult(success=True, message="Calculation for accounting closing balance is correct.")
    else:
        return ValidatorResult(success=False, message=f"Incorrect. Expected a value close to {expected}, but got {value}.")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Validator for taxation closing balance in 2019."""
    value = runtime.retrieve("q3_tax_closing_balance_2019")
    expected = CORRECT_ANSWERS["q3"]
    if value == expected:
        return ValidatorResult(success=True, message="Calculation for tax closing balance is correct.")
    else:
        return ValidatorResult(success=False, message=f"Incorrect. Expected a value close to {expected}, but got {value}.")

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Validator for the impact of the tax rate change."""
    value = runtime.retrieve("q4_tax_rate_change_impact")
    expected = CORRECT_ANSWERS["q4"]
    if abs(value) == expected:
        return ValidatorResult(success=True, message="Calculation for tax rate change impact is correct.")
    else:
        return ValidatorResult(success=False, message=f"Incorrect. Expected an impact of ~{expected}, but got {abs(value)}.")

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Validator for the deferred tax balance in 2028."""
    value = runtime.retrieve("q5_deferred_tax_balance_2028_value")
    type = runtime.retrieve("q5_deferred_tax_balance_2028_type")
    expected_value = CORRECT_ANSWERS["q5"]
    expected_type = CORRECT_ANSWERS["q5_type"]
    type_ok = isinstance(type, str) and type.lower() == expected_type.lower()
    if abs(value) and type_ok:
        return ValidatorResult(success=True, message="Deferred tax balance calculation is correct.")
    else:
        return ValidatorResult(success=False, message=f"Incorrect. Expected {expected_value} of {expected_type}, but got {abs(value)} of {type}.")

# ==============================================================================
# Framework Variable Definitions
# ==============================================================================

variables = [
    # Input Parameters
    Variable(
        name="real_capex_schedule", 
        value=real_capex_schedule, 
        description="A dictionary of real capex amounts in 2012 $m by year, example: {2012: 10, 2013: 64}, you should get the real capex schedule in this variable"
    ),
    Variable(
        name="inflation_rate", 
        value=inflation_rate, 
        description="The annual inflation rate, example: 0.03, you should get the real inflation rate in this variable"),
    Variable(
        name="accounting_depreciation_life", 
        value=accounting_depreciation_life, 
        description="The asset life in years for straight-line accounting depreciation, example: 12, you should get the accounting depreciation life in this variable"
    ),
    Variable(
        name="tax_depreciation_rate", 
        value=tax_depreciation_rate, 
        description="The rate for diminishing value tax depreciation, example: 0.40, you should get the tax depreciation rate in this variable"
    ),
    Variable(
        name="tax_rate_pre_2016", 
        value=tax_rate_pre_2016, 
        description="The tax rate before 2016, example: 0.30, you should get the tax rate before 2016 in this variable"),
    Variable(
        name="tax_rate_post_2016", 
        value=tax_rate_post_2016, 
        description="The tax rate from 2016 onwards, example: 0.28, you should get the tax rate from 2016 onwards in this variable"),

    # Variables to store answers
    Variable(
        name="q1_total_nominal_capex", 
        value=0, 
        description="Store the total nominal capex for 2012 through 2021 (inclusive)."),
    Variable(
        name="q2_accounting_closing_balance_2017", 
        value=0, 
        description="Store the closing balance for the accounting asset schedule in 2017."
    ),
    Variable(
        name="q3_tax_closing_balance_2019", 
        value=0, 
        description="Store the closing balance for the taxation asset schedule 2019."
    ),
    Variable(
        name="q4_tax_rate_change_impact", 
        value=0, 
        description="Store the impact on the opening deferred tax balance for 2016 from the change in tax rate."),
    Variable(
        name="q5_deferred_tax_balance_2028_value", 
        value=0, 
        description="Store the numeric value of the deferred tax balance as at 2028."),
    Variable(
        name="q5_deferred_tax_balance_2028_type", 
        value="", 
        description="Store the type of the deferred tax balance (Asset or Liability) as at 2028."),
]

validators = {
    "validate_q1": validate_q1,
    "validate_q2": validate_q2,
    "validate_q3": validate_q3,
    "validate_q4": validate_q4,
    "validate_q5": validate_q5,
}

# Usage example
if __name__ == "__main__":
    pass