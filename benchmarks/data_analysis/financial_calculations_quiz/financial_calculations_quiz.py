"""
This scenario is a quiz focused on common, self-contained financial and
accounting calculation problems.

QUESTIONS & ANSWERS
1.  If an $18,000 asset is depreciated using the diminishing balance (double
    declining) method, and assuming an effective life of 5 years, which is
    closest to the asset's book value after 3 years?
    [Answer: ~$4000]

2.  $10,000 is deposited in an account...How much more will the account balance
    be at the end of the year if interest is compounded monthly versus if
    interest is compounded quarterly?
    [Answer: ~$5]

3.  A loan facility with a $30m facility limit...incurs a commitment fee of 1%
    per annum on the daily undrawn balance. Drawdowns of $10m occur on each of
    1 February, 1 March and 1 April 2013...The total commitment fee incurred
    is closest to:
    [Answer: ~$50,000]
"""

from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn
from typing import List
from core.types import ToolCall
import math


# Inputs for Question 1
q1_asset_cost = 18000
q1_asset_life = 5

# Inputs for Question 3
q3_principal_deposit = 10000
q3_annual_interest_rate = 0.075

# Inputs for Question 7
q7_facility_limit = 30000000
q7_commitment_fee_rate = 0.01
q7_drawdowns = {
    "2013-02-01": 10000000,
    "2013-03-01": 10000000,
    "2013-04-01": 10000000
}

tools = []
TOLERANCE = 0.05 # 5% tolerance for multiple-choice approximations

CORRECT_ANSWERS = {
    "q1": 4000,
    "q3": 5.0,
    "q7": 50000,
}

# --- Individual Validator Functions ---

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Validates the depreciation calculation."""
    value = runtime.retrieve("q1_depreciation_answer")
    expected = CORRECT_ANSWERS["q1"]
    if isinstance(value, (int, float)) and math.isclose(value, expected, rel_tol=TOLERANCE):
        return ValidatorResult(success=True, message="Q1: Depreciation calculation is correct.")
    return ValidatorResult(success=False, message=f"Q1: Incorrect. Expected a value close to {expected}, but got '{value}'.")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Validates the interest compounding calculation."""
    value = runtime.retrieve("q3_compounding_answer")
    expected = CORRECT_ANSWERS["q3"]
    if isinstance(value, (int, float)) and math.isclose(value, expected, rel_tol=TOLERANCE):
        return ValidatorResult(success=True, message="Q3: Interest compounding calculation is correct.")
    return ValidatorResult(success=False, message=f"Q3: Incorrect. Expected a value close to {expected}, but got '{value}'.")

def validate_q7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Validates the commitment fee calculation."""
    value = runtime.retrieve("q7_fee_answer")
    expected = CORRECT_ANSWERS["q7"]
    if isinstance(value, (int, float)) and math.isclose(value, expected, rel_tol=TOLERANCE):
        return ValidatorResult(success=True, message="Q7: Commitment fee calculation is correct.")
    return ValidatorResult(success=False, message=f"Q7: Incorrect. Expected a value close to {expected}, but got '{value}'.")


variables = [
    # Input parameters
    Variable(
        name="q1_asset_cost", 
        value=q1_asset_cost,
        description="The cost of the asset for question 1, example: 18000, you should get the asset cost in this variable."
    ),
    Variable(
        name="q1_asset_life", 
        value=q1_asset_life,
        description="The effective life of the asset for question 1, example: 5, you should get the asset life in this variable."
    ),
    Variable(
        name="q3_principal_deposit", 
        value=q3_principal_deposit,
        description="The principal deposit for question 3, example: 10000, you should get the principal deposit in this variable."
    ),
    Variable(
        name="q3_annual_interest_rate", 
        value=q3_annual_interest_rate,
        description="The annual interest rate for question 3, example: 0.075, you should get the annual interest rate in this variable."
    ),
    Variable(
        name="q7_facility_limit", 
        value=q7_facility_limit,
        description="The facility limit for question 7, example: 30000000, you should get the facility limit in this variable."
    ),
    Variable(
        name="q7_commitment_fee_rate", 
        value=q7_commitment_fee_rate, 
        description="The commitment fee rate for question 7, example: 0.01, you should get the commitment fee rate in this variable."
    ),
    Variable(
        name="q7_drawdowns",
        value=q7_drawdowns,
        description="The drawdowns for question 7, example: {'2013-02-01': 10000000, '2013-03-01': 10000000, '2013-04-01': 10000000}, you should get the drawdowns in this variable."
    ),

    # Output answer placeholders
    Variable(
        name="q1_depreciation_answer", 
        value=0.0, 
        description="Store the asset's book value after 3 years for question 1."
    ),
    Variable(
        name="q3_compounding_answer", 
        value=0.0, 
        description="Store the extra amount earned from monthly vs quarterly compounding for question 3."
    ),
    Variable(
        name="q7_fee_answer", 
        value=0.0, 
        description="Store the total commitment fee incurred for question 7."),
]

validators = {
    "validate_q1": validate_q1,
    "validate_q3": validate_q3,
    "validate_q7": validate_q7,
}

if __name__ == "__main__":
    pass