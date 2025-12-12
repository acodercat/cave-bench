"""
This scenario is based on analyzing AAPL (Apple Inc.) stock data for the year 2024.
The goal is to use pandas and financial analysis techniques to calculate returns,
volatility, moving averages, and technical indicators.

Dataset Structure:
aapl_2024.csv contains multi-level headers similar to yfinance output:
- Date: Index
- Close: Closing price
- High: Daily high
- Low: Daily low
- Open: Opening price
- Volume: Trading volume

The data covers 2024-01-01 to 2024-12-31.

Question 1: Calculate the mean Close price for AAPL in 2024.

Question 2: Find the highest and lowest prices in 2024.

Question 3: Calculate average daily percentage returns based on the 'Close' price (in percentage).

Question 4: Resample the data to find the average 'Close' price for each month.

Question 5: Identify the month with the highest total trading volume.

Question 6: Calculate the absolute price change (Last Close - First Open) for each quarter.

Question 7: Calculate a 20-day Simple Moving Average (SMA) for the 'Close' price.

Question 8: Calculate a 20-day Exponential Moving Average (EMA) for the 'Close' price.

Question 9: Calculate the 20-day rolling annualized volatility (standard deviation of daily returns * sqrt(252)) for the 'Close' price.

Question 10: Identify dates where the daily return (absolute value) was greater than 4% for the 'Close' price.

Question 11: Calculate the correlation coefficient between Daily Volume and the absolute magnitude of Daily Returns for the 'Close' price.

Question 12: Calculate the Maximum Drawdown (MDD) of the Close price up to 2024-11-25.

Question 13: Identify dates where the 20-day SMA crosses above or below the 50-day SMA for the 'Close' price.

Question 14: Calculate the 14-day RSI (Relative Strength Index) for the 'Close' price using Wilder's Smoothing method.

Question 15: Simulate a "Buy and Hold" strategy: Calculate the value of a $10,000 investment made on day 1 over time.
"""

from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn
from typing import List
from core.types import ToolCall
import pandas as pd
import numpy as np

# Load the AAPL dataset
# Handling the specific header structure provided (Price/Ticker layers)
try:
    aapl_df = pd.read_csv("benchmarks/data_analysis/Finance/AAPL/AAPL_20240101_20241231.csv", parse_dates=True, index_col=0)
except Exception as e:
    print(f"Error loading AAPL dataset: {e}")
    aapl_df = None

tools = []

# Correct answers
correct_answers = {
    "q1": 205.67,  # Mean close price for AAPL in 2024
    "q2_high": 258.93,  # Highest price in 2024
    "q2_low": 162.75,  # Lowest price in 2024
    "q3": 0.1345,  # Average daily percentage returns for AAPL in 2024
    "q4_jan": 185.97, # Average monthly close price for January 2024
    "q4_dec": 248.15,  # Average monthly close price for December 2024
    "q5_month": ['2024-06', 'June'],  # Month with highest total trading volume for AAPL in 2024
    "q5_volume": 1717952100,  # Total trading volume for the month with highest total trading volume for AAPL in 2024
    "q6_q1": -15.31,  # Absolute price change for AAPL in 2024Q1
    "q6_q4": 22.83,  # Absolute price change for AAPL in 2024Q4
    "q7_20241231": 248.15,  # 20-day Simple Moving Average for AAPL on 2024-12-31
    "q8_20241231": 247.81,  # 20-day Exponential Moving Average for AAPL on 2024-12-31
    "q9_20241231": 0.1629,  # 20-day rolling annualized volatility for AAPL on 2024-12-31
    "q10": ['2024-03-21', '2024-04-11', '2024-05-03', '2024-06-11', '2024-08-05'],  # Dates with daily return greater than 2% for AAPL in 2024
    "q11": 0.3993,  # Correlation between Daily Volume and the absolute magnitude of Daily Returns for the 'Close' price in 2024
    "q12": 15.35,  # Maximum Drawdown for AAPL up to 2024-11-25
    "q13_golden_cross": ['2024-05-08', '2024-08-27', '2024-12-03'],  # Dates where the 20-day SMA crosses above the 50-day SMA for AAPL in 2024
    "q13_death_cross": ['2024-08-20', '2024-11-19'],  # Dates where the 20-day SMA crosses below the 50-day SMA for AAPL in 2024
    "q14_1230": 60.23,  # 14-day RSI for AAPL on 2024-12-30
    "q15_value": 13651.99,  # Value of a $10,000 investment made on day 1 over time for AAPL in 2024
    "q15_returns": 36.52,  # Returns of a $10,000 investment made on day 1 over time for AAPL in 2024
}
# --- Validators ---

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    mean_close = runtime.get_variable_value("mean_close_price")
    
    # Check type
    if not isinstance(mean_close, float):
        return ValidatorResult(success=False, message=f"Q1: Incorrect type. Expected float, but got '{type(mean_close).__name__}'.")
    
    # Check value
    if round(mean_close, 2) == correct_answers["q1"]:
        return ValidatorResult(success=True, message="Q1: Mean close price is correct.")
    
    return ValidatorResult(success=False, message=f"Q1: Incorrect. Expected {correct_answers['q1']}, but got {round(mean_close, 2)}.")

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    year_high = runtime.get_variable_value("year_high")
    year_low = runtime.get_variable_value("year_low")
    
    # Check types
    if not isinstance(year_high, float):
        return ValidatorResult(success=False, message=f"Q2: year_high has incorrect type. Expected float, but got '{type(year_high).__name__}'.")
    if not isinstance(year_low, float):
        return ValidatorResult(success=False, message=f"Q2: year_low has incorrect type. Expected float, but got '{type(year_low).__name__}'.")
    
    # Check values
    high_value = round(year_high, 2)
    low_value = round(year_low, 2)
    high_correct = high_value == correct_answers["q2_high"]
    low_correct = low_value == correct_answers["q2_low"]
    
    if high_correct and low_correct:
        return ValidatorResult(success=True, message="Q2: Year high and low are correct.")
    elif high_correct and not low_correct:
        return ValidatorResult(success=False, message=f"Q2: year_low is incorrect. Expected {correct_answers['q2_low']}, but got {low_value}.")
    elif not high_correct and low_correct:
        return ValidatorResult(success=False, message=f"Q2: year_high is incorrect. Expected {correct_answers['q2_high']}, but got {high_value}.")
    else:
        return ValidatorResult(success=False, message=f"Q2: Both values incorrect. Expected high={correct_answers['q2_high']}, low={correct_answers['q2_low']}, but got high={high_value}, low={low_value}.")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    average_daily_returns = runtime.get_variable_value("average_daily_returns")
    
    # Check type
    if not isinstance(average_daily_returns, float):
        return ValidatorResult(success=False, message=f"Q3: Incorrect type. Expected float, but got '{type(average_daily_returns).__name__}'.")
    
    # Check value
    if round(average_daily_returns, 4) == correct_answers["q3"]:
        return ValidatorResult(success=True, message="Q3: Average daily returns are correct.")
    
    return ValidatorResult(success=False, message=f"Q3: Incorrect. Expected {correct_answers['q3']}, but got {round(average_daily_returns, 4)}.")

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    average_monthly_close = runtime.get_variable_value("average_monthly_close")
    
    # Check type
    if not isinstance(average_monthly_close, pd.Series):
        return ValidatorResult(success=False, message=f"Q4: Incorrect type. Expected pandas Series, but got '{type(average_monthly_close).__name__}'.")
    
    # Check length
    if len(average_monthly_close) < 12:
        return ValidatorResult(success=False, message=f"Q4: Incorrect length. Expected 12 monthly values, but got {len(average_monthly_close)}.")
    
    # Check values
    jan_value = round(average_monthly_close.iloc[0], 2)
    dec_value = round(average_monthly_close.iloc[11], 2)
    jan_correct = jan_value == correct_answers["q4_jan"]
    dec_correct = dec_value == correct_answers["q4_dec"]
    
    if jan_correct and dec_correct:
        return ValidatorResult(success=True, message="Q4: Average monthly close prices are correct.")
    elif jan_correct and not dec_correct:
        return ValidatorResult(success=False, message=f"Q4: December value is incorrect. Expected {correct_answers['q4_dec']}, but got {dec_value}.")
    elif not jan_correct and dec_correct:
        return ValidatorResult(success=False, message=f"Q4: January value is incorrect. Expected {correct_answers['q4_jan']}, but got {jan_value}.")
    else:
        return ValidatorResult(success=False, message=f"Q4: Both values incorrect. Expected Jan={correct_answers['q4_jan']}, Dec={correct_answers['q4_dec']}, but got Jan={jan_value}, Dec={dec_value}.")

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    highest_vol_month = runtime.get_variable_value("highest_vol_month")
    highest_vol_month_volume = runtime.get_variable_value("highest_vol_month_volume")
    
    # Check volume type
    if not isinstance(highest_vol_month_volume, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q5: highest_vol_month_volume has incorrect type. Expected int, but got '{type(highest_vol_month_volume).__name__}'.")
    
    # Check volume value
    volume_correct = highest_vol_month_volume == correct_answers["q5_volume"]
    
    # Check if month matches any of the accepted formats
    accepted_formats = correct_answers["q5_month"]  # ['2024-06', 'June']
    month_correct = False
    
    # Handle different month types
    if isinstance(highest_vol_month, str):
        # Check if it matches any string format: '2024-06' or 'June'
        month_correct = highest_vol_month in accepted_formats[:2]
    else:
        return ValidatorResult(success=False, message=f"Q5: highest_vol_month has unexpected type '{type(highest_vol_month).__name__}'. Expected str.")
    
    # Return results based on correctness
    if month_correct and volume_correct:
        return ValidatorResult(success=True, message="Q5: Highest volume month and volume are correct.")
    elif month_correct and not volume_correct:
        return ValidatorResult(success=False, message=f"Q5: Month is correct, but volume is incorrect. Expected {correct_answers['q5_volume']}, but got {highest_vol_month_volume}.")
    elif not month_correct and volume_correct:
        return ValidatorResult(success=False, message=f"Q5: Volume is correct, but month is incorrect. Expected one of {accepted_formats}, but got '{highest_vol_month}'.")
    else:
        return ValidatorResult(success=False, message=f"Q5: Both values incorrect. Expected month=one of {accepted_formats}, volume={correct_answers['q5_volume']}, but got month='{highest_vol_month}', volume={highest_vol_month_volume}.")

def validate_q6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    quarterly_perf = runtime.get_variable_value("quarterly_performance")

    # Check type
    if not isinstance(quarterly_perf, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q6: Incorrect type. Expected pandas DataFrame, but got '{type(quarterly_perf).__name__}'.")
    
    # Check if price_change column exists
    if 'price_change' not in quarterly_perf.columns:
        return ValidatorResult(success=False, message=f"Q6: Missing 'price_change' column. Found columns: {quarterly_perf.columns.tolist()}.")
    
    # Check if we have at least 4 quarters
    if len(quarterly_perf) < 4:
        return ValidatorResult(success=False, message=f"Q6: Expected 4 quarters, but got {len(quarterly_perf)}.")
    
    # Get Q1 and Q4 values (first and last rows)
    try:
        q1_value = round(quarterly_perf['price_change'].iloc[0], 2)
        q4_value = round(quarterly_perf['price_change'].iloc[-1], 2)
    except (KeyError, IndexError) as e:
        return ValidatorResult(success=False, message=f"Q6: Error accessing quarterly data. Error: {e}")
    
    # Check values
    q1_correct = q1_value == correct_answers["q6_q1"]
    q4_correct = q4_value == correct_answers["q6_q4"]
    
    if q1_correct and q4_correct:
        return ValidatorResult(success=True, message="Q6: Quarterly performance is correct.")
    elif q1_correct and not q4_correct:
        return ValidatorResult(success=False, message=f"Q6: Q4 value is incorrect. Expected {correct_answers['q6_q4']}, but got {q4_value}.")
    elif not q1_correct and q4_correct:
        return ValidatorResult(success=False, message=f"Q6: Q1 value is incorrect. Expected {correct_answers['q6_q1']}, but got {q1_value}.")
    else:
        return ValidatorResult(success=False, message=f"Q6: Both values incorrect. Expected Q1={correct_answers['q6_q1']}, Q4={correct_answers['q6_q4']}, but got Q1={q1_value}, Q4={q4_value}.")
def validate_q7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    sma_20 = runtime.get_variable_value("sma_20")

    # Check type
    if not isinstance(sma_20, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q7: Incorrect type. Expected pandas DataFrame, but got '{type(sma_20).__name__}'.")

    # Check if SMA_20 column exists
    if 'SMA_20' not in sma_20.columns:
        return ValidatorResult(success=False, message=f"Q7: Missing 'SMA_20' column. Found columns: {sma_20.columns.tolist()}.")

    # Check value (last row should be 2024-12-31)
    sma_value = round(sma_20['SMA_20'].iloc[-1], 2)
    sma_correct = sma_value == correct_answers["q7_20241231"]
    if sma_correct:
        return ValidatorResult(success=True, message="Q7: 20-day Simple Moving Average for AAPL on 2024-12-31 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q7: Incorrect. Expected {correct_answers['q7_20241231']}, but got {sma_value}.")

def validate_q8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    ema_20 = runtime.get_variable_value("ema_20")

    # Check type
    if not isinstance(ema_20, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q8: Incorrect type. Expected pandas DataFrame, but got '{type(ema_20).__name__}'.")
    
    # Check if EMA_20 column exists
    if 'EMA_20' not in ema_20.columns:
        return ValidatorResult(success=False, message=f"Q8: Missing 'EMA_20' column. Found columns: {ema_20.columns.tolist()}.")
    
    # Check value (last row should be 2024-12-31)
    ema_value = round(ema_20['EMA_20'].iloc[-1], 2)
    ema_correct = ema_value == correct_answers["q8_20241231"]
    if ema_correct:
        return ValidatorResult(success=True, message="Q8: 20-day Exponential Moving Average for AAPL on 2024-12-31 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q8: Incorrect. Expected {correct_answers['q8_20241231']}, but got {ema_value}.")

def validate_q9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    rolling_vol = runtime.get_variable_value("rolling_volatility")

    # Check type
    if not isinstance(rolling_vol, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q9: Incorrect type. Expected pandas DataFrame, but got '{type(rolling_vol).__name__}'.")
    
    # Check if Vol_20_ann column exists
    if 'Vol_20_ann' not in rolling_vol.columns:
        return ValidatorResult(success=False, message=f"Q9: Missing 'Vol_20_ann' column. Found columns: {rolling_vol.columns.tolist()}.")
    
    # Check value (last row should be 2024-12-31)
    rolling_vol_value = round(rolling_vol['Vol_20_ann'].iloc[-1], 4)
    rolling_vol_correct = rolling_vol_value == correct_answers["q9_20241231"]
    if rolling_vol_correct:
        return ValidatorResult(success=True, message="Q9: 20-day rolling annualized volatility for AAPL on 2024-12-31 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q9: Incorrect. Expected {correct_answers['q9_20241231']}, but got {rolling_vol_value}.")

def validate_q10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    big_move_days = runtime.get_variable_value("big_move_days")

    # Check type
    if not isinstance(big_move_days, list):
        return ValidatorResult(success=False, message=f"Q10: Incorrect type. Expected list, but got '{type(big_move_days).__name__}'.")

    # Check values
    big_move_days_correct = all(date in correct_answers["q10"] for date in big_move_days)
    if big_move_days_correct:
        return ValidatorResult(success=True, message="Q10: High volatility days identified are correct.")
    else:
        return ValidatorResult(success=False, message=f"Q10: Incorrect. Expected {correct_answers['q10']}, but got {big_move_days}.")

def validate_q11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    corr = runtime.get_variable_value("vol_return_corr")

    # Check type
    if not isinstance(corr, float):
        return ValidatorResult(success=False, message=f"Q11: Incorrect type. Expected float, but got '{type(corr).__name__}'.")

    # Check value
    corr_value = round(corr, 4)
    corr_correct = corr_value == correct_answers["q11"]
    if corr_correct:
        return ValidatorResult(success=True, message="Q11: Correlation between Daily Volume and the absolute magnitude of Daily Returns for the 'Close' price in 2024 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q11: Incorrect. Expected {correct_answers['q11']}, but got {corr_value}.")

def validate_q12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    max_dd = runtime.get_variable_value("max_drawdown")

    # Check type
    if not isinstance(max_dd, float):
        return ValidatorResult(success=False, message=f"Q12: Incorrect type. Expected float, but got '{type(max_dd).__name__}'.")
        
    # Check value
    max_dd_value = abs(round(max_dd, 2))
    max_dd_correct = max_dd_value == correct_answers["q12"]
    if max_dd_correct:
        return ValidatorResult(success=True, message="Q12: Maximum Drawdown for AAPL up to 2024-11-25 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q12: Incorrect. Expected {correct_answers['q12']}, but got {max_dd_value}.")

def validate_q13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    golden_cross_dates = runtime.get_variable_value("golden_cross_dates")
    death_cross_dates = runtime.get_variable_value("death_cross_dates")

    # Check type
    if not isinstance(golden_cross_dates, list):
        return ValidatorResult(success=False, message=f"Q13: Incorrect type. Expected list, but got '{type(golden_cross).__name__}'.")
    if not isinstance(death_cross_dates, list):
        return ValidatorResult(success=False, message=f"Q13: Incorrect type. Expected list, but got '{type(death_cross).__name__}'.")

    # Check values
    golden_cross_correct = all(date in correct_answers["q13_golden_cross"] for date in golden_cross_dates)
    death_cross_correct = all(date in correct_answers["q13_death_cross"] for date in death_cross_dates)
    if golden_cross_correct and death_cross_correct:
        return ValidatorResult(success=True, message="Q13: Dates where the 20-day SMA crosses above or below the 50-day SMA for the 'Close' price in 2024 are correct.")
    elif golden_cross_correct and not death_cross_correct:
        return ValidatorResult(success=False, message=f"Q13: Death cross dates are incorrect. Expected {correct_answers['q13_death_cross']}, but got {death_cross_dates}.")
    elif not golden_cross_correct and death_cross_correct:
        return ValidatorResult(success=False, message=f"Q13: Golden cross dates are incorrect. Expected {correct_answers['q13_golden_cross']}, but got {golden_cross_dates}.")
    else:
        return ValidatorResult(success=False, message=f"Q13: Both values incorrect. Expected golden_cross={correct_answers['q13_golden_cross']}, death_cross={correct_answers['q13_death_cross']}, but got golden_cross={golden_cross_dates}, death_cross={death_cross_dates}.")

def validate_q14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    rsi = runtime.get_variable_value("rsi_14")

    # Check type
    if not isinstance(rsi, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q14: Incorrect type. Expected pandas DataFrame, but got '{type(rsi).__name__}'.")

    # Check if RSI_14 column exists
    if 'RSI_14' not in rsi.columns:
        return ValidatorResult(success=False, message=f"Q14: Missing 'RSI_14' column. Found columns: {rsi.columns.tolist()}.")

    # Check value (last row should be 2024-12-30)
    rsi_1230_value = round(rsi['RSI_14'].iloc[-1], 2)
    rsi_1230_correct = rsi_1230_value == correct_answers["q14_1230"]
    if rsi_1230_correct:
        return ValidatorResult(success=True, message="Q14: 14-day RSI for AAPL on 2024-12-30 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q14: Incorrect. Expected {correct_answers['q14_1230']}, but got {rsi_1230_value}.")
def validate_q15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    portfolio_value = runtime.get_variable_value("portfolio_value")
    portfolio_returns = runtime.get_variable_value("portfolio_returns")

    # Check type
    if not isinstance(portfolio_value, float):
        return ValidatorResult(success=False, message=f"Q15: Incorrect type. Expected float, but got '{type(portfolio_value).__name__}'.")
    if not isinstance(portfolio_returns, float):
        return ValidatorResult(success=False, message=f"Q15: Incorrect type. Expected float, but got '{type(portfolio_returns).__name__}'.")

    # Check value
    portfolio_value_value = round(portfolio_value, 2)
    portfolio_returns_value = round(portfolio_returns, 2)
    portfolio_value_correct = portfolio_value_value == correct_answers["q15_value"]
    portfolio_returns_correct = portfolio_returns_value == correct_answers["q15_returns"]
    if portfolio_value_correct and portfolio_returns_correct:
        return ValidatorResult(success=True, message="Q15: Value of a $10,000 investment made on day 1 over time for AAPL in 2024 is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q15: Incorrect. Expected {correct_answers['q15_value']}, but got {portfolio_value_value}.")
        return ValidatorResult(success=False, message=f"Q15: Incorrect. Expected {correct_answers['q15_returns']}, but got {portfolio_returns_value}.")

variables = [
    #Dataset
    Variable(
        name="aapl_df",
        value=aapl_df,
        description="""A pandas DataFrame from 'aapl_2024.csv'.
        Columns and their dtypes:
        - Date: object (string)
        - Open: float64
        - High: float64
        - Low: float64
        - Close: float64
        - Volume: int64
        example: pd.DataFrame(
            {
                "Date": ["2024-01-02"],
                "Open": [183.9032440185547],
                "High": [186.67705163638013],
                "Low": [182.16961614691823],
                "Close": [185.3991117688797],
                "Volume": [82488700]
            }
        )
        You should get the aapl_df from this variable to answer all the questions.
        """
    ),
    
    # Q1: Mean Close
    Variable(
        name="mean_close_price", 
        value=0.0, 
        description="Float. Store the arithmetic mean of the 'Close' price for AAPL in 2024 of question 1."
    ),

    # Q2: High / Low
    Variable(
        name="year_high", 
        value=0.0, 
        description="Float. Store the maximum value found in the 'High' price for AAPL in 2024 of question 2."
    ),
    Variable(
        name="year_low", 
        value=0.0, 
        description="Float. Store the minimum value found in the 'Low' price for AAPL in 2024 of question 2."
    ),

    # Q3: Daily Returns
    Variable(
        name="average_daily_returns", 
        value=0.0, 
        description="Float. Store the average of daily percentage returns (calculated based on 'Close' price) for AAPL in 2024 of question 3."
    ),

    # Q4: Monthly Resample
    Variable(
        name="average_monthly_close", 
        value=pd.Series(dtype=float), 
        description="""
        pandas Series. Store the average 'Close' price resampled by Month of question 4.
        - Index: PeriodIndex or DatetimeIndex (Monthly)
        - Values: Float (Average Close Price)
        """
    ),

    # Q5: Volume Analysis
    Variable(
        name="highest_vol_month", 
        value="", 
        description="String. Store the month (e.g., '2024-06', 'June') with the highest Total Volume of question 5."
    ),
    Variable(
        name="highest_vol_month_volume", 
        value=0, 
        description="Int. Store the total aggregated volume for the month with the highest trading volume for AAPL in 2024 of question 5."
    ),

    # Q6: Quarterly Performance
    Variable(
        name="quarterly_performance", 
        value=pd.DataFrame(), 
        description="""
        pandas DataFrame. Store the Absolute price change per quarter of question 6.
        - Index: DatetimeIndex (e.g., 2024-03-31, 2024-06-30, 2024-09-30, 2024-12-31) representing quarter end dates
        - Columns: ['price_change'] (Float, Last Close - First Open for each quarter)
        - Should contain 4 rows for the 4 quarters of 2024
        Example:
                    price_change
        Date                    
        2024-03-31    0.0
        2024-06-30    0.0
        2024-09-30    0.0
        2024-12-31    0.0
        """
    ),

    # Q7: SMA 20
    Variable(
        name="sma_20", 
        value=pd.DataFrame(), 
        description="""
        pandas DataFrame. Store the 20-day Simple Moving Average of question 7.
        - Index: DatetimeIndex (Date is the index, not a column)
        - Columns: ['SMA_20'] (Float)
        """
    ),

    # Q8: EMA 20
    Variable(
        name="ema_20", 
        value=pd.DataFrame(), 
        description="""
        pandas DataFrame. Store the 20-day Exponential Moving Average of question 8.
        - Index: DatetimeIndex (Date is the index, not a column)
        - Columns: ['EMA_20'] (Float)
        """
    ),

    # Q9: Volatility
    Variable(
        name="rolling_volatility", 
        value=pd.DataFrame(), 
        description="""
        pandas DataFrame. Store the 20-day Rolling Annualized Volatility of question 9.
        Calculation: Rolling Std Dev of Daily Returns * sqrt(252).
        - Index: DatetimeIndex (Date is the index, not a column)
        - Columns: ['Vol_20_ann'] (Float)
        """
    ),

    # Q10: Big Moves
    Variable(
        name="big_move_days", 
        value=[], 
        description="List of Strings. Store the Dates (YYYY-MM-DD) where the absolute daily return of 'Close' was > 4% of question 10."
    ),

    # Q11: Correlation
    Variable(
        name="vol_return_corr", 
        value=0.0, 
        description="Float. Store the Pearson correlation coefficient between Daily Volume and Absolute Daily Returns of question 11."
    ),

    # Q12: Max Drawdown
    Variable(
        name="max_drawdown", 
        value=0.0, 
        description="Float. Store the Maximum Drawdown (MDD) of the 'Close' price for AAPL up to 2024-11-25 (in percentage) of question 12."
    ),

    # Q13: Cross Over
    Variable(
        name="golden_cross_dates", 
        value=[], 
        description="List of Strings. Store the Dates (YYYY-MM-DD) where 20-day SMA crossed ABOVE 50-day SMA of question 13."
    ),
    Variable(
        name="death_cross_dates", 
        value=[], 
        description="List of Strings. Store the Dates (YYYY-MM-DD) where 20-day SMA crossed BELOW 50-day SMA of question 13."
    ),

    # Q14: RSI
    Variable(
        name="rsi_14", 
        value=pd.DataFrame(), 
        description="""
        pandas DataFrame. Store the 14-day Relative Strength Index of question 14.
        - Index: DatetimeIndex (Date is the index, not a column)
        - Columns: ['RSI_14'] (Float)
        """
    ),

    # Q15: Strategy
    Variable(
        name="portfolio_value", 
        value=0.0, 
        description="Float. Store the final value of a $10,000 investment made on the first day and held until the last day of question 15."
    ),
    Variable(
        name="portfolio_returns", 
        value=0.0, 
        description="Float. Store the percentage return of the portfolio at the end of the year of question 15."
    )
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