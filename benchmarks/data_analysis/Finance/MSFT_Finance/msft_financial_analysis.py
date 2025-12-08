"""
This scenario is based on analyzing Microsoft (MSFT) financial data.
The goal is to use pandas to perform financial analysis using three key financial statements:
1. Balance Sheet (Assets, Liabilities, Equity)
2. Income Statement (Revenue, Expenses, Profits)
3. Cash Flow Statement (Operating, Investing, Financing activities)

Dataset Structure:
The variables 'msft_balance_sheet', 'msft_income_statement', and 'msft_cash_flow' are pandas DataFrames provided by yfinance.
- Index: Date (e.g., '2022-06-30', '2023-06-30')
- Columns: Financial metrics (e.g., 'Total Assets', 'Net Income')
- Values: Financial amounts (Floats)

Data Range: Last 4 fiscal years (2022-2025) columns.

Question 1: Calculate the Year-over-Year (YoY) growth rate of 'Total Assets' for 2025.

Question 2: Identify the fiscal year with the highest YoY growth rate for 'Total Revenue' and provide that growth rate.

Question 3: Calculate the 'Working Capital' (Current Assets - Current Liabilities) for 2024.

Question 4: Calculate the 'Gross Margin' (Gross Profit / Total Revenue) for ALL available years. Store the result in a DataFrame named 'gross_margin' with columns ['Year', 'Gross_Margin_%'].

Question 5: Calculate the 'Operating Margin' (Operating Income / Total Revenue) for 2025.

Question 6: Calculate the 'Quick Ratio' ((Current Assets - Inventory) / Current Liabilities) for ALL years. Identify the fiscal year with the LOWEST Quick Ratio and provide that value.

Question 7: Calculate 'Free Cash Flow' (Operating Cash Flow + Capital Expenditure) for 2025. Note: CapEx is usually negative.

Question 8: [Cross-Table] Calculate 'Asset Turnover' (Total Revenue / Total Assets) for 2024.

Question 9: [Cross-Table] Calculate 'Inventory Turnover' (Reconciled Cost of Revenue / Inventory) for 2024.

Question 10: [Cross-Table] Calculate 'Return on Assets' (Net Income / Total Assets) for ALL available years.

Question 11: [Cross-Table] Calculate 'Return on Equity' (Net Income / Stockholders Equity) for ALL available years.

Question 12: [Cross-Table] Calculate 'Days Sales Outstanding' (DSO) for the most recent year. Formula: (Net Receivables / Total Revenue) * 365.

Question 13: [Cross-Table] Calculate the 'FCF Conversion Ratio' (Free Cash Flow / Net Income) for 2025.

Question 14: [Cross-Table] Perform a Dupont Analysis for 2025. Calculate: Net Margin, Asset Turnover, and Financial Leverage (Assets/Equity).

Question 15: [Cross-Table] Calculate the 'Rule of 40' score for 2025. Formula: (Revenue Growth Rate + Free Cash Flow Margin).
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd
import numpy as np

# Load the MSFT financial data
try:
    msft_balance_sheet = pd.read_csv("benchmarks/data_analysis/Finance/MSFT/msft_balance_sheet.csv")
except Exception as e:
    print(f"Error loading MSFT balance sheet dataset: {e}")
    msft_balance_sheet = None
try:
    msft_income_statement = pd.read_csv("benchmarks/data_analysis/Finance/MSFT/msft_income_statement.csv")
except Exception as e:
    print(f"Error loading MSFT income statement dataset: {e}")
    msft_income_statement = None
try:
    msft_cash_flow = pd.read_csv("benchmarks/data_analysis/Finance/MSFT/msft_cash_flow.csv")
except Exception as e:
    print(f"Error loading MSFT cash flow dataset: {e}")
    msft_cash_flow = None

tools = []

correct_answers = {
    "q1": 20.86,
    "q2_year": 2024,
    "q2_growth_rate": 15.67,
    "q3": 34448000000.0,
    "q4_2025": 68.82,
    "q4_2024": 69.76,
    "q5": 45.62,
    "q6_year": 2024,
    "q6_ratio": 1.268,
    "q7": 71611000000.0,
    "q8": 47.86,
    "q9": 87.7,
    "q10_2022": 19.94,
    "q10_2023": 17.56,
    "q11_2022": 43.68,
    "q11_2023": 35.09,
    "q12": 90.6,
    "q13": 70.32,
    "q14_net_margin": 36.15,
    "q14_asset_turnover": 45.51,
    "q14_financial_leverage": 1.8022,
    "q15": 40.35,
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for YoY growth rate of Total Assets
    yoy_2025_growth_rate = runtime.get_variable("yoy_2025_growth_rate")

    # Check type
    if not isinstance(yoy_2025_growth_rate, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q1: yoy_2025_growth_rate has incorrect type. Expected float, but got '{type(yoy_2025_growth_rate).__name__}'.")

    # Normalize value: if > 1, assume it's percentage; if < 1, convert to percentage
    if abs(yoy_2025_growth_rate) > 1:
        # Already in percentage format (e.g., 20.86)
        normalized_value = round(yoy_2025_growth_rate, 2)
    else:
        # In decimal format (e.g., 0.2086), convert to percentage
        normalized_value = round(yoy_2025_growth_rate * 100, 2)
    
    # Check value
    if normalized_value == correct_answers["q1"]:
        return ValidatorResult(success=True, message="Q1: YoY growth rate of Total Assets for 2025 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q1: Incorrect. Expected {correct_answers['q1']}%, but got {normalized_value}%.")

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for YoY growth rate of Total Revenue
    highest_yoy_growth_rate_year = runtime.get_variable("highest_yoy_growth_rate_year")
    highest_yoy_growth_rate = runtime.get_variable("highest_yoy_growth_rate")

    # Check types
    if not isinstance(highest_yoy_growth_rate_year, int):
        return ValidatorResult(success=False, message=f"Q2: year has incorrect type. Expected int, but got '{type(highest_yoy_growth_rate_year).__name__}'.")
    if not isinstance(highest_yoy_growth_rate, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q2: highest_yoy_growth_rate has incorrect type. Expected float, but got '{type(highest_yoy_growth_rate).__name__}'.")
    
    # Normalize value: if > 1, assume it's percentage; if < 1, convert to percentage
    if abs(highest_yoy_growth_rate) > 1:
        # Already in percentage format (e.g., 15.67)
        normalized_value = round(highest_yoy_growth_rate, 2)
    else:
        # In decimal format (e.g., 0.1567), convert to percentage
        normalized_value = round(highest_yoy_growth_rate * 100, 2)
    
    # Check value
    if highest_yoy_growth_rate_year == correct_answers["q2_year"] and normalized_value == correct_answers["q2_growth_rate"]:
        return ValidatorResult(success=True, message="Q2: The fiscal year with the highest YoY growth rate for Total Revenue is correct.")
    elif highest_yoy_growth_rate_year == correct_answers["q2_year"] and not normalized_value == correct_answers["q2_growth_rate"]:
        return ValidatorResult(success=False, message=f"Q2: Incorrect. Expected {correct_answers['q2_year']} with {correct_answers['q2_growth_rate']}%, but got {highest_yoy_growth_rate_year} with {normalized_value}%.")
    elif not highest_yoy_growth_rate_year == correct_answers["q2_year"] and normalized_value == correct_answers["q2_growth_rate"]:
        return ValidatorResult(success=False, message=f"Q2: Incorrect. Expected {correct_answers['q2_year']} with {correct_answers['q2_growth_rate']}%, but got {highest_yoy_growth_rate_year} with {normalized_value}%.")
    else:
        return ValidatorResult(success=False, message=f"Q2: Both values incorrect. Expected year={correct_answers['q2_year']}, growth rate={correct_answers['q2_growth_rate']}%, but got year={highest_yoy_growth_rate_year}, growth rate={normalized_value}%.")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Working Capital
    working_capital_2024 = runtime.get_variable("working_capital_2024")

    # Check type
    if not isinstance(working_capital_2024, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q3: working_capital_2024 has incorrect type. Expected float, but got '{type(working_capital_2024).__name__}'.")

    # Check value
    if round(working_capital_2024, 1) == correct_answers["q3"]:
        return ValidatorResult(success=True, message="Q3: Working Capital for 2024 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q3: Incorrect. Expected {correct_answers['q3']}, but got {working_capital_2024}.")

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Gross Margin
    gross_margin = runtime.get_variable("gross_margin")
    
    # Check type
    if not isinstance(gross_margin, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q4: gross_margin has incorrect type. Expected pandas DataFrame, but got '{type(gross_margin).__name__}'.")
    
    # Check if required columns exist
    if 'Year' not in gross_margin.columns:
        return ValidatorResult(success=False, message=f"Q4: Missing 'Year' column. Found columns: {gross_margin.columns.tolist()}.")
    if 'Gross_Margin_%' not in gross_margin.columns:
        return ValidatorResult(success=False, message=f"Q4: Missing 'Gross_Margin_%' column. Found columns: {gross_margin.columns.tolist()}.")
    
    # Get 2024 and 2025 values
    try:
        gross_margin_2024 = round(gross_margin[gross_margin['Year'] == 2024]['Gross_Margin_%'].iloc[0], 2)
        gross_margin_2025 = round(gross_margin[gross_margin['Year'] == 2025]['Gross_Margin_%'].iloc[0], 2)
    except (IndexError, KeyError) as e:
        return ValidatorResult(success=False, message=f"Q4: Error accessing data for years 2024 or 2025. Error: {e}")
    
    # Check values
    correct_2024 = gross_margin_2024 == correct_answers["q4_2024"]
    correct_2025 = gross_margin_2025 == correct_answers["q4_2025"]
    
    if correct_2024 and correct_2025:
        return ValidatorResult(success=True, message="Q4: Gross Margin for 2024 and 2025 is correct.")
    elif correct_2024 and not correct_2025:
        return ValidatorResult(success=False, message=f"Q4: 2025 Gross Margin is incorrect. Expected {correct_answers['q4_2025']}%, but got {gross_margin_2025}%.")
    elif not correct_2024 and correct_2025:
        return ValidatorResult(success=False, message=f"Q4: 2024 Gross Margin is incorrect. Expected {correct_answers['q4_2024']}%, but got {gross_margin_2024}%.")
    else:
        return ValidatorResult(success=False, message=f"Q4: Both values incorrect. Expected 2024={correct_answers['q4_2024']}%, 2025={correct_answers['q4_2025']}%, but got 2024={gross_margin_2024}%, 2025={gross_margin_2025}%.")

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Operating Margin
    operating_margin = runtime.get_variable("operating_margin")
    
    # Check type
    if not isinstance(operating_margin, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q5: operating_margin has incorrect type. Expected float, but got '{type(operating_margin).__name__}'.")
    
    # Normalize value: if > 1, assume it's percentage; if < 1, convert to percentage
    if abs(operating_margin) > 1:
        # Already in percentage format (e.g., 45.62)
        normalized_value = round(operating_margin, 2)
    else:
        # In decimal format (e.g., 0.4562), convert to percentage
        normalized_value = round(operating_margin * 100, 2)
    
    # Check value
    if normalized_value == correct_answers["q5"]:
        return ValidatorResult(success=True, message="Q5: Operating Margin for 2025 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q5: Incorrect. Expected {correct_answers['q5']}%, but got {normalized_value}%.")

def validate_q6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Quick Ratio
    quick_ratio = runtime.get_variable("quick_ratio")
    
    # Check type
    if not isinstance(quick_ratio, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q6: quick_ratio has incorrect type. Expected pandas DataFrame, but got '{type(quick_ratio).__name__}'.")
    
    # Check if required columns exist
    if 'Year' not in quick_ratio.columns:
        return ValidatorResult(success=False, message=f"Q6: Missing 'Year' column. Found columns: {quick_ratio.columns.tolist()}.")
    if 'Quick_Ratio' not in quick_ratio.columns:
        return ValidatorResult(success=False, message=f"Q6: Missing 'Quick_Ratio' column. Found columns: {quick_ratio.columns.tolist()}.")
    
    # Get the lowest Quick Ratio
    lowest_quick_ratio = quick_ratio['Quick_Ratio'].min()
    lowest_quick_ratio_year = quick_ratio['Year'].loc[quick_ratio['Quick_Ratio'] == lowest_quick_ratio].iloc[0]
    ratio_correct = round(lowest_quick_ratio, 3) == correct_answers["q6_ratio"]
    year_correct = lowest_quick_ratio_year == correct_answers["q6_year"]
    
    # Check value
    if ratio_correct and year_correct:
        return ValidatorResult(success=True, message="Q6: The fiscal year with the lowest Quick Ratio is correct.")
    elif ratio_correct and not year_correct:
        return ValidatorResult(success=False, message=f"Q6: Incorrect. Expected {correct_answers['q6_year']} with {correct_answers['q6_ratio']}, but got {lowest_quick_ratio_year} with {lowest_quick_ratio}.")
    elif not ratio_correct and year_correct:
        return ValidatorResult(success=False, message=f"Q6: Incorrect. Expected {correct_answers['q6_year']} with {correct_answers['q6_ratio']}, but got {lowest_quick_ratio_year} with {lowest_quick_ratio}.")
    else:
        return ValidatorResult(success=False, message=f"Q6: Both values incorrect. Expected year={correct_answers['q6_year']}, ratio={correct_answers['q6_ratio']}, but got year={lowest_quick_ratio_year}, ratio={lowest_quick_ratio}.")

def validate_q7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Free Cash Flow
    free_cash_flow_2025 = runtime.get_variable("free_cash_flow_2025")
    
    # Check type
    if not isinstance(free_cash_flow_2025, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q7: free_cash_flow_2025 has incorrect type. Expected float, but got '{type(free_cash_flow_2025).__name__}'.")
    
    # Check value
    if round(free_cash_flow_2025, 1) == correct_answers["q7"]:
        return ValidatorResult(success=True, message="Q7: Free Cash Flow for 2025 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q7: Incorrect. Expected {correct_answers['q7']}, but got {free_cash_flow_2025}.")

def validate_q8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Asset Turnover
    asset_turnover_2024 = runtime.get_variable("asset_turnover_2024")
    
    # Check type
    if not isinstance(asset_turnover_2024, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q8: asset_turnover_2024 has incorrect type. Expected float, but got '{type(asset_turnover_2024).__name__}'.")
    
    # Normalize value: if > 1, assume it's percentage; if < 1, convert to percentage
    if abs(asset_turnover_2024) > 1:
        # Already in percentage format (e.g., 45.51)
        normalized_value = round(asset_turnover_2024, 2)
    else:
        # In decimal format (e.g., 0.4551), convert to percentage
        normalized_value = round(asset_turnover_2024 * 100, 2)
    # Check value
    if normalized_value == correct_answers["q8"]:
        return ValidatorResult(success=True, message="Q8: Asset Turnover for 2024 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q8: Incorrect. Expected {correct_answers['q8']}%, but got {normalized_value}%.")

def validate_q9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Inventory Turnover
    inventory_turnover_2024 = runtime.get_variable("inventory_turnover_2024")
    
    # Check type
    if not isinstance(inventory_turnover_2024, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q9: inventory_turnover_2024 has incorrect type. Expected float, but got '{type(inventory_turnover_2024).__name__}'.")
    
    # Check value
    if round(inventory_turnover_2024, 1) == correct_answers["q9"]:
        return ValidatorResult(success=True, message="Q9: Inventory Turnover for 2024 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q9: Incorrect. Expected {correct_answers['q9']}, but got {inventory_turnover_2024}.")

def validate_q10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Return on Assets
    roa = runtime.get_variable("roa")
    
    # Check types
    if not isinstance(roa, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q10: roa has incorrect type. Expected pandas DataFrame, but got '{type(roa).__name__}'.")
    
    # Check if required columns exist
    if 'Year' not in roa.columns:
        return ValidatorResult(success=False, message=f"Q10: Missing 'Year' column. Found columns: {roa.columns.tolist()}.")
    if 'RoA_%' not in roa.columns:
        return ValidatorResult(success=False, message=f"Q10: Missing 'RoA_%' column. Found columns: {roa.columns.tolist()}.")
    
    # Get 2022 and 2023 values
    try:
        roa_2022 = round(roa[roa['Year'] == 2022]['RoA_%'].iloc[0], 2)
        roa_2023 = round(roa[roa['Year'] == 2023]['RoA_%'].iloc[0], 2)
    except (IndexError, KeyError) as e:
        return ValidatorResult(success=False, message=f"Q10: Error accessing data for years 2022 or 2023. Error: {e}")
    
    # Check values
    correct_2022 = roa_2022 == correct_answers["q10_2022"]
    correct_2023 = roa_2023 == correct_answers["q10_2023"]
    
    if correct_2022 and correct_2023:
        return ValidatorResult(success=True, message="Q10: Return on Assets for 2022 and 2023 is correct.")
    elif correct_2022 and not correct_2023:
        return ValidatorResult(success=False, message=f"Q10: 2023 Return on Assets is incorrect. Expected {correct_answers['q10_2023']}%, but got {roa_2023}%.")
    elif not correct_2022 and correct_2023:
        return ValidatorResult(success=False, message=f"Q10: 2022 Return on Assets is incorrect. Expected {correct_answers['q10_2022']}%, but got {roa_2022}%.")
    else:
        return ValidatorResult(success=False, message=f"Q10: Both values incorrect. Expected 2022={correct_answers['q10_2022']}%, 2023={correct_answers['q10_2023']}%, but got 2022={roa_2022}%, 2023={roa_2023}%.")

def validate_q11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Return on Equity
    roe = runtime.get_variable("roe")
    
    # Check types
    if not isinstance(roe, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q11: roe has incorrect type. Expected pandas DataFrame, but got '{type(roe).__name__}'.")
    if 'Year' not in roe.columns:
        return ValidatorResult(success=False, message=f"Q11: Missing 'Year' column. Found columns: {roe.columns.tolist()}.")
    if 'RoE_%' not in roe.columns:
        return ValidatorResult(success=False, message=f"Q11: Missing 'RoE_%' column. Found columns: {roe.columns.tolist()}.")
    
    # Get 2022 and 2023 values
    try:
        roe_2022 = round(roe[roe['Year'] == 2022]['RoE_%'].iloc[0], 2)
        roe_2023 = round(roe[roe['Year'] == 2023]['RoE_%'].iloc[0], 2)
    except (IndexError, KeyError) as e:
        return ValidatorResult(success=False, message=f"Q11: Error accessing data for years 2022 or 2023. Error: {e}")
    
    # Check values
    correct_2022 = roe_2022 == correct_answers["q11_2022"]
    correct_2023 = roe_2023 == correct_answers["q11_2023"]
    
    if correct_2022 and correct_2023:
        return ValidatorResult(success=True, message="Q11: Return on Equity for 2022 and 2023 is correct.")
    elif correct_2022 and not correct_2023:
        return ValidatorResult(success=False, message=f"Q11: 2023 Return on Equity is incorrect. Expected {correct_answers['q11_2023']}%, but got {roe_2023}%.")
    elif not correct_2022 and correct_2023:
        return ValidatorResult(success=False, message=f"Q11: 2022 Return on Equity is incorrect. Expected {correct_answers['q11_2022']}%, but got {roe_2022}%.")
    else:
        return ValidatorResult(success=False, message=f"Q11: Both values incorrect. Expected 2022={correct_answers['q11_2022']}%, 2023={correct_answers['q11_2023']}%, but got 2022={roe_2022}%, 2023={roe_2023}%.")

def validate_q12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Days Sales Outstanding
    dso_recent = runtime.get_variable("dso_recent")
    
    # Check types
    if not isinstance(dso_recent, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q12: dso_recent has incorrect type. Expected float, but got '{type(dso_recent).__name__}'.")
    
    # Check value
    if round(dso_recent, 1) == correct_answers["q12"]:
        return ValidatorResult(success=True, message="Q12: Days Sales Outstanding for the most recent year is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q12: Incorrect. Expected {correct_answers['q12']}, but got {dso_recent}.")

def validate_q13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for FCF Conversion Ratio
    fcf_conversion_ratio = runtime.get_variable("fcf_conversion_ratio")
    
    # Check types
    if not isinstance(fcf_conversion_ratio, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q13: fcf_conversion_ratio has incorrect type. Expected float, but got '{type(fcf_conversion_ratio).__name__}'.")
    
    # Normalize value: if > 1, assume it's percentage; if < 1, convert to percentage
    if abs(fcf_conversion_ratio) > 1:
        # Already in percentage format (e.g., 70.32)
        normalized_value = round(fcf_conversion_ratio, 2)
    else:
        # In decimal format (e.g., 0.7032), convert to percentage
        normalized_value = round(fcf_conversion_ratio * 100, 2)
    # Check value
    if normalized_value == correct_answers["q13"]:
        return ValidatorResult(success=True, message="Q13: FCF Conversion Ratio for 2025 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q13: Incorrect. Expected {correct_answers['q13']}%, but got {normalized_value}%.")

def validate_q14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Dupont Analysis
    net_margin_2025 = runtime.get_variable("net_margin_2025")
    asset_turnover_2025 = runtime.get_variable("asset_turnover_2025")
    financial_leverage_2025 = runtime.get_variable("financial_leverage_2025")
    
    # Check types
    if not isinstance(net_margin_2025, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q14: net_margin_2025 has incorrect type. Expected float, but got '{type(net_margin_2025).__name__}'.")
    if not isinstance(asset_turnover_2025, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q14: asset_turnover_2025 has incorrect type. Expected float, but got '{type(asset_turnover_2025).__name__}'.")
    if not isinstance(financial_leverage_2025, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q14: financial_leverage_2025 has incorrect type. Expected float, but got '{type(financial_leverage_2025).__name__}'.")
    
    # Normalize values: if > 1, assume it's percentage; if < 1, convert to percentage
    if abs(net_margin_2025) > 1:
        # Already in percentage format (e.g., 36.15)
        normalized_net_margin = round(net_margin_2025, 2)
    else:
        # In decimal format (e.g., 0.3615), convert to percentage
        normalized_net_margin = round(net_margin_2025 * 100, 2)
    if abs(asset_turnover_2025) > 1:
        # Already in percentage format (e.g., 45.51)
        normalized_asset_turnover = round(asset_turnover_2025, 2)
    else:
        # In decimal format (e.g., 0.4551), convert to percentage
        normalized_asset_turnover = round(asset_turnover_2025 * 100, 2)

    net_margin_correct = normalized_net_margin == correct_answers["q14_net_margin"]
    asset_turnover_correct = normalized_asset_turnover == correct_answers["q14_asset_turnover"]
    financial_leverage_correct = round(financial_leverage_2025, 4) == correct_answers["q14_financial_leverage"]
    
    if net_margin_correct and asset_turnover_correct and financial_leverage_correct:
        return ValidatorResult(success=True, message="Q14: Dupont Analysis for 2025 is correct.")
    elif net_margin_correct and not asset_turnover_correct and not financial_leverage_correct:
        return ValidatorResult(success=False, message=f"Q14: Incorrect. Expected net margin={correct_answers['q14_net_margin']}%, asset turnover={correct_answers['q14_asset_turnover']}%, financial leverage={correct_answers['q14_financial_leverage']}%, but got net margin={normalized_net_margin}%, asset turnover={normalized_asset_turnover}%, financial leverage={financial_leverage_2025}%.")
    elif not net_margin_correct and asset_turnover_correct and not financial_leverage_correct:
        return ValidatorResult(success=False, message=f"Q14: Incorrect. Expected net margin={correct_answers['q14_net_margin']}%, asset turnover={correct_answers['q14_asset_turnover']}%, financial leverage={correct_answers['q14_financial_leverage']}%, but got net margin={normalized_net_margin}%, asset turnover={normalized_asset_turnover}%, financial leverage={financial_leverage_2025}%.")
    elif not net_margin_correct and not asset_turnover_correct and financial_leverage_correct:
        return ValidatorResult(success=False, message=f"Q14: Incorrect. Expected net margin={correct_answers['q14_net_margin']}%, asset turnover={correct_answers['q14_asset_turnover']}%, financial leverage={correct_answers['q14_financial_leverage']}%, but got net margin={normalized_net_margin}%, asset turnover={normalized_asset_turnover}%, financial leverage={financial_leverage_2025}%.")
    else:
        return ValidatorResult(success=False, message=f"Q14: All values incorrect. Expected net margin={correct_answers['q14_net_margin']}%, asset turnover={correct_answers['q14_asset_turnover']}%, financial leverage={correct_answers['q14_financial_leverage']}%, but got net margin={normalized_net_margin}%, asset turnover={normalized_asset_turnover}%, financial leverage={financial_leverage_2025}%.")

def validate_q15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Rule of 40
    rule_of_40 = runtime.get_variable("rule_of_40")
    
    # Check types
    if not isinstance(rule_of_40, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q15: rule_of_40 has incorrect type. Expected float, but got '{type(rule_of_40).__name__}'.")
    
    # Check value
    if round(rule_of_40, 2) == correct_answers["q15"]:
        return ValidatorResult(success=True, message="Q15: Rule of 40 for 2025 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q15: Incorrect. Expected {correct_answers['q15']}, but got {round(rule_of_40, 2)}.")

variables = [
    # 1. Balance Sheet
    Variable(
        name="msft_balance_sheet",
        value=msft_balance_sheet,
        description="""A pandas DataFrame from 'msft_balance_sheet.csv' containing Microsoft's balance sheet data.
        Columns and their dtypes:
        - Date: object (string)
        - Ordinary Shares Number: float64
        - Share Issued: float64
        - Net Debt: float64
        - Total Debt: float64
        - Tangible Book Value: float64
        - Invested Capital: float64
        - Working Capital: float64
        - Net Tangible Assets: float64
        - Capital Lease Obligations: float64
        - Common Stock Equity: float64
        - Total Capitalization: float64
        - Total Equity Gross Minority Interest: float64
        - Stockholders Equity: float64
        - Gains Losses Not Affecting Retained Earnings: float64
        - Other Equity Adjustments: float64
        - Retained Earnings: float64
        - Capital Stock: float64
        - Common Stock: float64
        - Total Liabilities Net Minority Interest: float64
        - Total Non Current Liabilities Net Minority Interest: float64
        - Other Non Current Liabilities: float64
        - Tradeand Other Payables Non Current: float64
        - Non Current Deferred Liabilities: float64
        - Non Current Deferred Revenue: float64
        - Non Current Deferred Taxes Liabilities: float64
        - Long Term Debt And Capital Lease Obligation: float64
        - Long Term Capital Lease Obligation: float64
        - Long Term Debt: float64
        - Current Liabilities: float64
        - Other Current Liabilities: float64
        - Current Deferred Liabilities: float64
        - Current Deferred Revenue: float64
        - Current Debt And Capital Lease Obligation: float64
        - Current Debt: float64
        - Other Current Borrowings: float64
        - Commercial Paper: float64
        - Pensionand Other Post Retirement Benefit Plans Current: float64
        - Payables And Accrued Expenses: float64
        - Payables: float64
        - Total Tax Payable: float64
        - Income Tax Payable: float64
        - Accounts Payable: float64
        - Total Assets: float64
        - Total Non Current Assets: float64
        - Other Non Current Assets: float64
        - Financial Assets: float64
        - Investments And Advances: float64
        - Investmentin Financial Assets: float64
        - Available For Sale Securities: float64
        - Long Term Equity Investment: float64
        - Goodwill And Other Intangible Assets: float64
        - Other Intangible Assets: float64
        - Goodwill: float64
        - Net PPE: float64
        - Accumulated Depreciation: float64
        - Gross PPE: float64
        - Leases: float64
        - Other Properties: float64
        - Machinery Furniture Equipment: float64
        - Buildings And Improvements: float64
        - Land And Improvements: float64
        - Properties: float64
        - Current Assets: float64
        - Other Current Assets: float64
        - Hedging Assets Current: float64
        - Inventory: float64
        - Finished Goods: float64
        - Work In Process: float64
        - Raw Materials: float64
        - Receivables: float64
        - Accounts Receivable: float64
        - Allowance For Doubtful Accounts Receivable: float64
        - Gross Accounts Receivable: float64
        - Cash Cash Equivalents And Short Term Investments: float64
        - Other Short Term Investments: float64
        - Cash And Cash Equivalents: float64
        - Cash Equivalents: float64
        - Cash Financial: float64
        
        Example: pd.DataFrame({
            "Date": ["2025-06-30"],
            "Ordinary Shares Number": [7434000000.0],
            "Share Issued": [7434000000.0],
            "Net Debt": [12909000000.0],
            "Total Debt": [60588000000.0],
            "Tangible Book Value": [201366000000.0],
            "Invested Capital": [386630000000.0],
            "Working Capital": [49913000000.0],
            "Net Tangible Assets": [201366000000.0],
            "Capital Lease Obligations": [17437000000.0],
            "Common Stock Equity": [343479000000.0],
            "Total Capitalization": [383631000000.0],
            "Total Equity Gross Minority Interest": [343479000000.0],
            "Stockholders Equity": [343479000000.0],
            "Gains Losses Not Affecting Retained Earnings": [-3347000000.0],
            "Other Equity Adjustments": [-3347000000.0],
            "Retained Earnings": [237731000000.0],
            "Capital Stock": [109095000000.0],
            "Common Stock": [109095000000.0],
            "Total Liabilities Net Minority Interest": [275524000000.0],
            "Total Non Current Liabilities Net Minority Interest": [134306000000.0],
            "Other Non Current Liabilities": [45186000000.0],
            "Tradeand Other Payables Non Current": [25986000000.0],
            "Non Current Deferred Liabilities": [5545000000.0],
            "Non Current Deferred Revenue": [2710000000.0],
            "Non Current Deferred Taxes Liabilities": [2835000000.0],
            "Long Term Debt And Capital Lease Obligation": [57589000000.0],
            "Long Term Capital Lease Obligation": [17437000000.0],
            "Long Term Debt": [40152000000.0],
            "Current Liabilities": [141218000000.0],
            "Other Current Liabilities": [25020000000.0],
            "Current Deferred Liabilities": [64555000000.0],
            "Current Deferred Revenue": [64555000000.0],
            "Current Debt And Capital Lease Obligation": [2999000000.0],
            "Current Debt": [2999000000.0],
            "Other Current Borrowings": [2999000000.0],
            "Commercial Paper": [0.0],
            "Pensionand Other Post Retirement Benefit Plans Current": [13709000000.0],
            "Payables And Accrued Expenses": [34935000000.0],
            "Payables": [34935000000.0],
            "Total Tax Payable": [7211000000.0],
            "Income Tax Payable": [7211000000.0],
            "Accounts Payable": [27724000000.0],
            "Total Assets": [619003000000.0],
            "Total Non Current Assets": [427872000000.0],
            "Other Non Current Assets": [40565000000.0],
            "Financial Assets": [272000000.0],
            "Investments And Advances": [15133000000.0],
            "Investmentin Financial Assets": [2460000000.0],
            "Available For Sale Securities": [2460000000.0],
            "Long Term Equity Investment": [12673000000.0],
            "Goodwill And Other Intangible Assets": [142113000000.0],
            "Other Intangible Assets": [22604000000.0],
            "Goodwill": [119509000000.0],
            "Net PPE": [229789000000.0],
            "Accumulated Depreciation": [-93653000000.0],
            "Gross PPE": [323442000000.0],
            "Leases": [12117000000.0],
            "Other Properties": [24823000000.0],
            "Machinery Furniture Equipment": [139243000000.0],
            "Buildings And Improvements": [137921000000.0],
            "Land And Improvements": [9338000000.0],
            "Properties": [0.0],
            "Current Assets": [191131000000.0],
            "Other Current Assets": [25723000000.0],
            "Hedging Assets Current": [10000000.0],
            "Inventory": [938000000.0],
            "Finished Goods": [None],
            "Work In Process": [None],
            "Raw Materials": [None],
            "Receivables": [69905000000.0],
            "Accounts Receivable": [69905000000.0],
            "Allowance For Doubtful Accounts Receivable": [-944000000.0],
            "Gross Accounts Receivable": [70849000000.0],
            "Cash Cash Equivalents And Short Term Investments": [94555000000.0],
            "Other Short Term Investments": [64313000000.0],
            "Cash And Cash Equivalents": [30242000000.0],
            "Cash Equivalents": [18531000000.0],
            "Cash Financial": [11711000000.0]
        })
        """
    ),
    
    # 2. Income Statement
    Variable(
        name="msft_income_statement",
        value=msft_income_statement,
        description="""A pandas DataFrame from 'msft_income_statement.csv' containing Microsoft's income statement data.
        Columns and their dtypes:
        - Date: object (string)
        - Tax Effect Of Unusual Items: float64
        - Tax Rate For Calcs: float64
        - Normalized EBITDA: float64
        - Total Unusual Items: float64
        - Total Unusual Items Excluding Goodwill: float64
        - Net Income From Continuing Operation Net Minority Interest: float64
        - Reconciled Depreciation: float64
        - Reconciled Cost Of Revenue: float64
        - EBITDA: float64
        - EBIT: float64
        - Net Interest Income: float64
        - Interest Expense: float64
        - Interest Income: float64
        - Normalized Income: float64
        - Net Income From Continuing And Discontinued Operation: float64
        - Total Expenses: float64
        - Total Operating Income As Reported: float64
        - Diluted Average Shares: float64
        - Basic Average Shares: float64
        - Diluted EPS: float64
        - Basic EPS: float64
        - Diluted NI Availto Com Stockholders: float64
        - Net Income Common Stockholders: float64
        - Net Income: float64
        - Net Income Including Noncontrolling Interests: float64
        - Net Income Continuous Operations: float64
        - Tax Provision: float64
        - Pretax Income: float64
        - Other Income Expense: float64
        - Other Non Operating Income Expenses: float64
        - Special Income Charges: float64
        - Write Off: float64
        - Gain On Sale Of Security: float64
        - Net Non Operating Interest Income Expense: float64
        - Interest Expense Non Operating: float64
        - Interest Income Non Operating: float64
        - Operating Income: float64
        - Operating Expense: float64
        - Research And Development: float64
        - Selling General And Administration: float64
        - Selling And Marketing Expense: float64
        - General And Administrative Expense: float64
        - Other Gand A: float64
        - Gross Profit: float64
        - Cost Of Revenue: float64
        - Total Revenue: float64
        - Operating Revenue: float64
        
        Example: pd.DataFrame({
            "Date": ["2025-06-30"],
            "Tax Effect Of Unusual Items": [-77088000.0],
            "Tax Rate For Calcs": [0.176],
            "Normalized EBITDA": [160603000000.0],
            "Total Unusual Items": [-438000000.0],
            "Total Unusual Items Excluding Goodwill": [-438000000.0],
            "Net Income From Continuing Operation Net Minority Interest": [101832000000.0],
            "Reconciled Depreciation": [34153000000.0],
            "Reconciled Cost Of Revenue": [87831000000.0],
            "EBITDA": [160165000000.0],
            "EBIT": [126012000000.0],
            "Net Interest Income": [262000000.0],
            "Interest Expense": [2385000000.0],
            "Interest Income": [2647000000.0],
            "Normalized Income": [102192912000.0],
            "Net Income From Continuing And Discontinued Operation": [101832000000.0],
            "Total Expenses": [153196000000.0],
            "Total Operating Income As Reported": [128528000000.0],
            "Diluted Average Shares": [7465000000.0],
            "Basic Average Shares": [7433000000.0],
            "Diluted EPS": [13.64],
            "Basic EPS": [13.7],
            "Diluted NI Availto Com Stockholders": [101832000000.0],
            "Net Income Common Stockholders": [101832000000.0],
            "Net Income": [101832000000.0],
            "Net Income Including Noncontrolling Interests": [101832000000.0],
            "Net Income Continuous Operations": [101832000000.0],
            "Tax Provision": [21795000000.0],
            "Pretax Income": [123627000000.0],
            "Other Income Expense": [-5163000000.0],
            "Other Non Operating Income Expenses": [-4725000000.0],
            "Special Income Charges": [-943000000.0],
            "Write Off": [943000000.0],
            "Gain On Sale Of Security": [505000000.0],
            "Net Non Operating Interest Income Expense": [262000000.0],
            "Interest Expense Non Operating": [2385000000.0],
            "Interest Income Non Operating": [2647000000.0],
            "Operating Income": [128528000000.0],
            "Operating Expense": [65365000000.0],
            "Research And Development": [32488000000.0],
            "Selling General And Administration": [32877000000.0],
            "Selling And Marketing Expense": [25654000000.0],
            "General And Administrative Expense": [7223000000.0],
            "Other Gand A": [7223000000.0],
            "Gross Profit": [193893000000.0],
            "Cost Of Revenue": [87831000000.0],
            "Total Revenue": [281724000000.0],
            "Operating Revenue": [281724000000.0]
        })
        """
    ),

    # 3. Cash Flow
    Variable(
        name="msft_cash_flow",
        value=msft_cash_flow,
        description="""A pandas DataFrame from 'msft_cash_flow.csv' containing Microsoft's cash flow statement data.
        Columns and their dtypes:
        - Date: object (string)
        - Free Cash Flow: float64
        - Repurchase Of Capital Stock: float64
        - Repayment Of Debt: float64
        - Issuance Of Debt: float64
        - Issuance Of Capital Stock: float64
        - Capital Expenditure: float64
        - End Cash Position: float64
        - Beginning Cash Position: float64
        - Effect Of Exchange Rate Changes: float64
        - Changes In Cash: float64
        - Financing Cash Flow: float64
        - Cash Flow From Continuing Financing Activities: float64
        - Net Other Financing Charges: float64
        - Cash Dividends Paid: float64
        - Common Stock Dividend Paid: float64
        - Net Common Stock Issuance: float64
        - Common Stock Payments: float64
        - Common Stock Issuance: float64
        - Net Issuance Payments Of Debt: float64
        - Net Short Term Debt Issuance: float64
        - Short Term Debt Issuance: float64
        - Net Long Term Debt Issuance: float64
        - Long Term Debt Payments: float64
        - Long Term Debt Issuance: float64
        - Investing Cash Flow: float64
        - Cash Flow From Continuing Investing Activities: float64
        - Net Other Investing Changes: float64
        - Net Investment Purchase And Sale: float64
        - Sale Of Investment: float64
        - Purchase Of Investment: float64
        - Net Business Purchase And Sale: float64
        - Purchase Of Business: float64
        - Net PPE Purchase And Sale: float64
        - Purchase Of PPE: float64
        - Operating Cash Flow: float64
        - Cash Flow From Continuing Operating Activities: float64
        - Change In Working Capital: float64
        - Change In Other Working Capital: float64
        - Change In Other Current Liabilities: float64
        - Change In Other Current Assets: float64
        - Change In Payables And Accrued Expense: float64
        - Change In Payable: float64
        - Change In Account Payable: float64
        - Change In Tax Payable: float64
        - Change In Income Tax Payable: float64
        - Change In Inventory: float64
        - Change In Receivables: float64
        - Changes In Account Receivables: float64
        - Stock Based Compensation: float64
        - Unrealized Gain Loss On Investment Securities: float64
        - Asset Impairment Charge: float64
        - Deferred Tax: float64
        - Deferred Income Tax: float64
        - Depreciation Amortization Depletion: float64
        - Depreciation And Amortization: float64
        - Depreciation: float64
        - Operating Gains Losses: float64
        - Gain Loss On Investment Securities: float64
        - Net Income From Continuing Operations: float64
        
        Example: pd.DataFrame({
            "Date": ["2025-06-30"],
            "Free Cash Flow": [71611000000.0],
            "Repurchase Of Capital Stock": [-18420000000.0],
            "Repayment Of Debt": [-3216000000.0],
            "Issuance Of Debt": [0.0],
            "Issuance Of Capital Stock": [2056000000.0],
            "Capital Expenditure": [-64551000000.0],
            "End Cash Position": [30242000000.0],
            "Beginning Cash Position": [18315000000.0],
            "Effect Of Exchange Rate Changes": [63000000.0],
            "Changes In Cash": [11864000000.0],
            "Financing Cash Flow": [-51699000000.0],
            "Cash Flow From Continuing Financing Activities": [-51699000000.0],
            "Net Other Financing Charges": [-2291000000.0],
            "Cash Dividends Paid": [-24082000000.0],
            "Common Stock Dividend Paid": [-24082000000.0],
            "Net Common Stock Issuance": [-16364000000.0],
            "Common Stock Payments": [-18420000000.0],
            "Common Stock Issuance": [2056000000.0],
            "Net Issuance Payments Of Debt": [-8962000000.0],
            "Net Short Term Debt Issuance": [-5746000000.0],
            "Short Term Debt Issuance": [None],
            "Net Long Term Debt Issuance": [-3216000000.0],
            "Long Term Debt Payments": [-3216000000.0],
            "Long Term Debt Issuance": [0.0],
            "Investing Cash Flow": [-72599000000.0],
            "Cash Flow From Continuing Investing Activities": [-72599000000.0],
            "Net Other Investing Changes": [2317000000.0],
            "Net Investment Purchase And Sale": [-4387000000.0],
            "Sale Of Investment": [25388000000.0],
            "Purchase Of Investment": [-29775000000.0],
            "Net Business Purchase And Sale": [-5978000000.0],
            "Purchase Of Business": [-5978000000.0],
            "Net PPE Purchase And Sale": [-64551000000.0],
            "Purchase Of PPE": [-64551000000.0],
            "Operating Cash Flow": [136162000000.0],
            "Cash Flow From Continuing Operating Activities": [136162000000.0],
            "Change In Working Capital": [-5350000000.0],
            "Change In Other Working Capital": [5400000000.0],
            "Change In Other Current Liabilities": [4947000000.0],
            "Change In Other Current Assets": [-5994000000.0],
            "Change In Payables And Accrued Expense": [569000000.0],
            "Change In Payable": [569000000.0],
            "Change In Account Payable": [569000000.0],
            "Change In Tax Payable": [None],
            "Change In Income Tax Payable": [None],
            "Change In Inventory": [309000000.0],
            "Change In Receivables": [-10581000000.0],
            "Changes In Account Receivables": [-10581000000.0],
            "Stock Based Compensation": [11974000000.0],
            "Unrealized Gain Loss On Investment Securities": [-536000000.0],
            "Asset Impairment Charge": [943000000.0],
            "Deferred Tax": [-7056000000.0],
            "Deferred Income Tax": [-7056000000.0],
            "Depreciation Amortization Depletion": [34153000000.0],
            "Depreciation And Amortization": [34153000000.0],
            "Depreciation": [34153000000.0],
            "Operating Gains Losses": [202000000.0],
            "Gain Loss On Investment Securities": [202000000.0],
            "Net Income From Continuing Operations": [101832000000.0]
        })
        """
    ),

    # --- Result Variables for Agents to Write To ---

    # Q1
    Variable(
        name="yoy_2025_growth_rate", 
        value=0.0, 
        description="Float. YoY growth rate of Total Assets for 2025."
    ),
    # Q2
    Variable(
        name="highest_yoy_growth_rate_year", 
        value=0, 
        description="Int. The fiscal year with the highest YoY growth rate for Total Revenue."
    ),
    Variable(
        name="highest_yoy_growth_rate", 
        value=0.0, 
        description="Float. The YoY growth rate for the fiscal year with the highest YoY growth rate for Total Revenue."
    ),
    # Q3
    Variable(
        name="working_capital_2024", 
        value=0.0, 
        description="Float. Working Capital (Current Assets - Current Liabilities) for 2024."
    ),
    # Q4
    Variable(
        name="gross_margin", 
        value=pd.DataFrame(),
        description="""
        pandas DataFrame representing the Gross Margin for all available years.
        - Index: Default integer index (0, 1, 2, 3...)
        - Columns: ['Year', 'Gross_Margin_%']
          - 'Year': Integer, fiscal year (e.g., 2022, 2023, 2024, 2025)
          - 'Gross_Margin_%': Float, gross margin percentage
        """
    ),
    # Q5
    Variable(
        name="operating_margin",
        value=0.0,
        description="Float. Operating Margin (Operating Income / Total Revenue) for 2025."
    ),
    # Q6
    Variable(
        name="quick_ratio", 
        value=pd.DataFrame(), 
        description="""
        pandas DataFrame representing the Quick Ratio for all available years.
        - Index: Default integer index (0, 1, 2, 3...)
        - Columns: ['Year', 'Quick_Ratio']
          - 'Year': Integer, fiscal year (e.g., 2025, 2024, 2023, 2022)
          - 'Quick_Ratio': Float, quick ratio
        """),
    # Q7
    Variable(
        name="free_cash_flow_2025", 
        value=0.0, 
        description="Float. Free Cash Flow (Operating Cash Flow + Capital Expenditure) for 2025."),
    # Q8
    Variable(
        name="asset_turnover_2024",
        value=0.0,
        description="Float. Asset Turnover (Total Revenue / Total Assets) for 2024."
    ),
    # Q9
    Variable(
        name="inventory_turnover_2024", 
        value=0.0, 
        description="Float. Inventory Turnover (Reconciled Cost of Revenue / Inventory) for 2024."),
    # Q10
    Variable(
        name="roa", 
        value=pd.DataFrame(), 
        description="""
        pandas DataFrame representing the Return on Assets for all available years.
        - Index: Default integer index (0, 1, 2, 3...)
        - Columns: ['Year', 'RoA_%']
          - 'Year': Integer, fiscal year (e.g., 2022, 2023, 2024, 2025)
          - 'RoA_%': Float, return on assets percentage
    """),
    # Q11
    Variable(
        name="roe", 
        value=pd.DataFrame(), 
        description="""
        pandas DataFrame representing the Return on Equity for all available years.
        - Index: Default integer index (0, 1, 2, 3...)
        - Columns: ['Year', 'RoE_%']
          - 'Year': Integer, fiscal year (e.g., 2022, 2023, 2024, 2025)
          - 'RoE_%': Float, return on equity percentage
    """),
    # Q12
    Variable(
        name="dso_recent",
        value=0.0,
        description="Float. Days Sales Outstanding (Net Receivables / Total Revenue) * 365 for the most recent year."
    ),
    # Q13
    Variable(
        name="fcf_conversion_ratio", 
        value=0.0, 
        description="Float. FCF Conversion Ratio (Free Cash Flow / Net Income) for 2025."),
    # Q14
    Variable(
        name="net_margin_2025",
        value=0.0,
        description="Float. Net Margin (Net Income / Total Revenue) for 2025."),
    Variable(
        name="asset_turnover_2025",
        value=0.0,
        description="Float. Asset Turnover (Total Revenue / Total Assets) for 2025."),
    Variable(
        name="financial_leverage_2025", 
        value=0.0, 
        description="Float. Financial Leverage (Total Assets / Stockholders Equity) for 2025."),
    # Q15
    Variable(
        name="rule_of_40",
        value=0.0,
        description="Float. Rule of 40 (Revenue Growth Rate + Free Cash Flow Margin) for 2025."),
]

validators = {
    "validate_q1": validate_q1,
    "validate_q2": validate_q2,
    "validate_q3": validate_q3,
    "validate_q4": validate_q4,
    "validate_q5": validate_q5,
    "validate_q6": validate_q6,
    "validate_q7": validate_q7,
    "validate_q8": validate_q8,
    "validate_q9": validate_q9,
    "validate_q10": validate_q10,
    "validate_q11": validate_q11,
    "validate_q12": validate_q12,
    "validate_q13": validate_q13,
    "validate_q14": validate_q14,
    "validate_q15": validate_q15,
}

if __name__ == "__main__":
    pass