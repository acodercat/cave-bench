"""
This scenario is based on analyzing the AirQualityUCI dataset.
The goal is to use pandas to perform data cleaning, time series analysis, and correlation analysis.

Dataset Structure:
- The variable 'air_quality_raw' is a pandas DataFrame.
- Columns include: Date, Time, CO(GT), PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC), etc.
- It may contain empty columns (parsing artifacts) and missing values marked as -200.

Question 1: Data Cleaning (Drop unnamed columns, Drop rows with missing Time/Date, Replace -200 with NaN, Interpolate).
Question 2: Daily Cycle Analysis (Hourly average of CO(GT)).
Question 3: Seasonal Analysis (Winter vs Summer NOx levels).
Question 4: Weather Impact (Correlation between T and PT08.S1).
Question 5: PCA Analysis (Identify feature with highest loading on PC1, excluding weather variables T, RH, AH).
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd
import numpy as np

# Load the Raw Data
try:
    air_quality_raw = pd.read_csv("benchmarks/data_analysis/Air_Quality/AirQualityUCI/AirQualityUCI.csv")
except Exception as e:
    print(f"Error loading AirQuality dataset: {e}")
    air_quality_raw = None

tools = []

correct_answers = {
    # Q1
    "q1_unnamed_columns": 0,
    "q1_200_values": 0,
    "q1_null_values": 0,
    "q1_no2_gt_9": 47.0,
    # Q2
    "q2_peak_hour": 19,
    "q2_peak_hour_concentration": 3.503,
    # Q3
    "q3_winter": 344.796,
    "q3_summer": 107.928,
    # Q4
    "q4": 0.0312,
    # Q5
    "q5_pc1_dominant": "PT08.S2(NMHC)"
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Data Cleaning
    cleaned_df = runtime.get_variable("air_quality_cleaned")

    if not isinstance(cleaned_df, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q1: Result is not a DataFrame. Got {type(cleaned_df)}.")

    # Check 1: Ensure Unnamed columns are dropped
    unnamed_columns = sum(1 for col in cleaned_df.columns if "Unnamed" in str(col))
    if unnamed_columns != correct_answers["q1_unnamed_columns"]:
        return ValidatorResult(success=False, message=f"Q1: The DataFrame still contains 'Unnamed' columns. Please drop columns with no proper header. Expected {correct_answers['q1_unnamed_columns']}, got {unnamed_columns}.")
    
    # Check 2: Check if -200 still exists
    _200_values = (cleaned_df == -200).any().any()
    if _200_values != correct_answers["q1_200_values"]:
        return ValidatorResult(success=False, message=f"Q1: Data still contains -200 values. They should be replaced with NaN. Expected {correct_answers['q1_200_values']}, got {_200_values}.")

    # Check 3: Check if null values are filled
    null_values = cleaned_df.isnull().sum().sum()
    if null_values != correct_answers["q1_null_values"]:
        return ValidatorResult(success=False, message=f"Q1: Data still contains null values. They should be filled. Expected {correct_answers['q1_null_values']}, got {null_values}.")

    # Check 4: Verify interpolation by checking NO2(GT) column at row 10 (index 9)
    if "NO2(GT)" in cleaned_df.columns:
        no2_gt_value = round(cleaned_df["NO2(GT)"].iloc[9], 1)
        if no2_gt_value != correct_answers["q1_no2_gt_9"]:
            return ValidatorResult(success=False, message=f"Q1: Interpolation verification failed. NO2(GT) at row 10 (index 9) should be {correct_answers['q1_no2_gt_9']}, got {no2_gt_value}.")
    else:
        return ValidatorResult(success=False, message="Q1: Column 'NO2(GT)' not found in cleaned DataFrame.")

    return ValidatorResult(success=True, message="Q1: Data cleaning (drop columns, replace values, interpolate) performed successfully.")

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Hourly Analysis
    hourly_avg = runtime.get_variable("hourly_co_avg")

    if not isinstance(hourly_avg, pd.Series):
        return ValidatorResult(success=False, message="Q2: Result should be a Series.")
    
    if len(hourly_avg) != 24:
        return ValidatorResult(success=False, message=f"Q2: Result should have 24 hours. Got {len(hourly_avg)}.")
    
    peak_hour = hourly_avg.idxmax()
    peak_hour_concentration = round(hourly_avg.max(), 3)
    if peak_hour != correct_answers["q2_peak_hour"] or peak_hour_concentration != correct_answers["q2_peak_hour_concentration"]:
        return ValidatorResult(success=False, message=f"Q2: Peak hour should be {correct_answers['q2_peak_hour']}, got {peak_hour}. Peak hour concentration should be {correct_answers['q2_peak_hour_concentration']}, got {peak_hour_concentration}.")
    return ValidatorResult(success=True, message=f"Q2: Correct. Peak hour is {peak_hour} with concentration {peak_hour_concentration}.")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Seasonal Analysis
    seasonal_nox = runtime.get_variable("seasonal_nox")

    if not isinstance(seasonal_nox, dict):
        return ValidatorResult(success=False, message="Q3: Result should be a dictionary.")
    
    if "Winter" not in seasonal_nox or "Summer" not in seasonal_nox:
        return ValidatorResult(success=False, message="Q3: Dictionary keys should be 'Winter' and 'Summer'.")

    winter_nox = round(seasonal_nox["Winter"], 3)
    summer_nox = round(seasonal_nox["Summer"], 3)
    if winter_nox != correct_answers["q3_winter"] or summer_nox != correct_answers["q3_summer"]:
        return ValidatorResult(success=False, message=f"Q3: Winter NOx should be {correct_answers['q3_winter']}, got {winter_nox}. Summer NOx should be {correct_answers['q3_summer']}, got {summer_nox}.")
    return ValidatorResult(success=True, message=f"Q3: Correct. Winter NOx is {winter_nox} and Summer NOx is {summer_nox}.")

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for Weather Correlation
    corr = runtime.get_variable("temp_sensor_corr")

    if not isinstance(corr, float):
        return ValidatorResult(success=False, message=f"Q4: Result should be a float. Got {type(corr)}.")

    if round(corr, 4) != correct_answers["q4"]:
        return ValidatorResult(success=False, message=f"Q4: Correlation coefficient should be {correct_answers['q4']}, got {round(corr, 4)}.")

    return ValidatorResult(success=True, message=f"Q4: Correct. Correlation coefficient is {correct_answers['q4']}.")

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation logic for PCA Analysis
    dominant_feature = runtime.get_variable("pc1_dominant_feature")

    if not isinstance(dominant_feature, str):
        return ValidatorResult(success=False, message=f"Q5: Result should be a string (column name). Got {type(dominant_feature)}.")

    # We expect the feature with the highest absolute loading on PC1.
    if dominant_feature != correct_answers["q5_pc1_dominant"]:
        return ValidatorResult(success=False, message=f"Q5: Identified '{dominant_feature}', but expected '{correct_answers['q5_pc1_dominant']}' as the feature with the highest absolute loading on PC1. Ensure data was standardized before PCA.")

    return ValidatorResult(success=True, message="Q5: Correctly identified the dominant feature from PCA.")

variables = [
    # Input Variable
    Variable(
        name="air_quality_raw",
        value=air_quality_raw,
        description="""Raw DataFrame of AirQualityUCI data. May contain empty 'Unnamed' columns and missing values marked as -200.
        Columns and their dtypes (in order):
        - Date: object (string, format: "M/D/YYYY" e.g., "3/10/2004")
        - Time: object (string, format: "HH:MM:SS" e.g., "18:00:00")
        - CO(GT): float64
        - PT08.S1(CO): float64
        - NMHC(GT): float64
        - C6H6(GT): float64
        - PT08.S2(NMHC): float64
        - NOx(GT): float64
        - PT08.S3(NOx): float64
        - NO2(GT): float64
        - PT08.S4(NO2): float64
        - PT08.S5(O3): float64
        - T: float64 (Temperature)
        - RH: float64 (Relative Humidity)
        - AH: float64 (Absolute Humidity)
        - Unnamed: 15: float64 (empty column, should be dropped)
        - Unnamed: 16: float64 (empty column, should be dropped)
        Missing values are represented as -200 and should be replaced with NaN before interpolation.
        Example: pd.DataFrame(data={
            "Date": ["3/10/2004"],
            "Time": ["18:00:00"],
            "CO(GT)": [2.6],
            "PT08.S1(CO)": [1360.0],
            "NMHC(GT)": [150.0],
            "C6H6(GT)": [11.9],
            "PT08.S2(NMHC)": [1046.0],
            "NOx(GT)": [166.0],
            "PT08.S3(NOx)": [1056.0],
            "NO2(GT)": [113.0],
            "PT08.S4(NO2)": [1692.0],
            "PT08.S5(O3)": [1268.0],
            "T": [13.6],
            "RH": [48.9],
            "AH": [0.7578],
        })
        You should use this variable to answer question 1.
        """
    ),
    
    # Output Variables
    Variable(
        name="air_quality_cleaned",
        value=pd.DataFrame(),
        description="""Store the DataFrame after dropping unnamed columns, dropping invalid rows, replacing -200 with NaN, and interpolating missing values of question 1.
        Columns and their dtypes (in order):
        - Date: object (string, format: "M/D/YYYY" e.g., "3/10/2004")
        - Time: object (string, format: "HH:MM:SS" e.g., "18:00:00")
        - CO(GT): float64
        - PT08.S1(CO): float64
        - NMHC(GT): float64
        - C6H6(GT): float64
        - PT08.S2(NMHC): float64
        - NOx(GT): float64
        - PT08.S3(NOx): float64
        - NO2(GT): float64
        - PT08.S4(NO2): float64
        - PT08.S5(O3): float64
        - T: float64 (Temperature)
        - RH: float64 (Relative Humidity)
        - AH: float64 (Absolute Humidity)
        Example: pd.DataFrame(data={
            "Date": ["3/10/2004"],
            "Time": ["18:00:00"],
            "CO(GT)": [2.6],
            "PT08.S1(CO)": [1360.0],
            "NMHC(GT)": [150.0],
            "C6H6(GT)": [11.9],
            "PT08.S2(NMHC)": [1046.0],
            "NOx(GT)": [166.0],
            "PT08.S3(NOx)": [1056.0],
            "NO2(GT)": [113.0],
            "PT08.S4(NO2)": [1692.0],
            "PT08.S5(O3)": [1268.0],
            "T": [13.6],
            "RH": [48.9],
            "AH": [0.7578],
        })
        You should use this variable to answer question 2,3,4 and 5.
        """
    ),
    Variable(
        name="hourly_co_avg",
        value=pd.Series(dtype=float),
        description="Store the Series containing average CO(GT) for each hour (0-23) of question 2."
    ),
    Variable(
        name="seasonal_nox",
        value={},
        description="Store the Dictionary with keys 'Winter' and 'Summer' containing average NOx(GT) levels of question 3."
    ),
    Variable(
        name="temp_sensor_corr",
        value=0.0,
        description="Store the Pearson correlation coefficient between Temperature (T) and PT08.S1(CO) of question 4."
    ),
    Variable(
        name="pc1_dominant_feature",
        value="",
        description="Store the name of the feature (column) that has the highest absolute loading on the first principal component (PC1) of question 5."
    ),
]

validators = {
    "validate_q1": validate_q1,
    "validate_q2": validate_q2,
    "validate_q3": validate_q3,
    "validate_q4": validate_q4,
    "validate_q5": validate_q5,
}

if __name__ == "__main__":
    pass