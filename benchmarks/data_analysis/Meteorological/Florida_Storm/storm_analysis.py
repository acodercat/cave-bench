"""
This scenario is based on analyzing NOAA Storm Data for Florida (2024/07/01-2025/07/31).
The goal is to use pandas and geospatial techniques to analyze storm patterns, social impact, and physical characteristics.

Dataset Structure:
The variable 'storm_data' is a pandas DataFrame.
- Columns: 
  - EVENT_ID: int64 (Unique identifier for each event)
  - CZ_NAME_STR: object (County Name)
  - BEGIN_LOCATION: object (Start location of the event)
  - BEGIN_DATE: object (Start date, format: "MM/DD/YYYY")
  - BEGIN_TIME: int64 (Start time in HHMM format)
  - EVENT_TYPE: object (Type of storm, e.g., "Thunderstorm Wind", "Heavy Rain")
  - MAGNITUDE: object (Physical intensity, may be NaN)
  - TOR_F_SCALE: object (Tornado scale, may be NaN)
  - DEATHS_DIRECT: int64 (Direct deaths)
  - INJURIES_DIRECT: int64 (Direct injuries)
  - DAMAGE_PROPERTY_NUM: int64 (Property damage amount)
  - DAMAGE_CROPS_NUM: int64 (Crops damage amount)
  - MAGNITUDE_TYPE: object (Type of magnitude measurement, may be NaN)
  - INJURIES_INDIRECT: int64 (Indirect injuries)
  - DEATHS_INDIRECT: int64 (Indirect deaths)
  - SOURCE: object (Data source, e.g., "ASOS", "Mesonet", "Public")
  - FLOOD_CAUSE: object (Flood cause, may be NaN)
  - TOR_LENGTH: object (Tornado length, may be NaN)
  - TOR_WIDTH: object (Tornado width, may be NaN)
  - BEGIN_RANGE: object (Begin range, may be numeric string or NaN)
  - BEGIN_AZIMUTH: object (Begin azimuth, e.g., "NE", "SW", may be NaN)
  - END_RANGE: object (End range, may be numeric string or NaN)
  - END_AZIMUTH: object (End azimuth, e.g., "NE", "SW", may be NaN)
  - END_LOCATION: object (End location of the event)
  - END_DATE: object (End date, format: "MM/DD/YYYY")
  - END_TIME: int64 (End time in HHMM format)
  - BEGIN_LAT: object (Latitude as string, e.g., "30.36")
  - BEGIN_LON: object (Longitude as string, e.g., "-85.8")
  - END_LAT: object (Latitude as string)
  - END_LON: object (Longitude as string)
  - EVENT_NARRATIVE: object (Text description of the event, may be NaN)

Question 1: [Complex Filtering & Aggregation] 
            Identify the 'Glass Jaw' county (high fragility). 
            Filter for 'Thunderstorm Wind' events where the wind MAGNITUDE was < 50 knots. 
            For counties with at least 2 such events, calculate the Average Property Damage.
            Identify the county with the highest average damage.

Question 2: [Time Series] 
            Analyze 'Flash Flood' events. 
            Calculate the average duration (in minutes) of Flash Floods that caused property damage > 0.

Question 3: [Statistical Analysis] 
            Verify the 'Pareto Principle' (80/20 rule) for Property Damage.
            Consider only events with damage > 0.
            Calculate the percentage of top events (by count) required to account for 80% of the Total Property Damage.
            Round up to the next integer number of events if the 80% threshold falls between events.
            Calculate the percentage as: (number of events needed / total events with damage > 0) * 100.
            (e.g., if the top 5 events out of 100 caused 80% of the loss, the answer is 5.0%)

Question 4: [Intensity Analysis] 
            Compare the economic impact of storm intensity levels.
            Compare 'Tropical Storm' events vs. 'Hurricane (Typhoon)' events.
            Calculate the Average Property Damage for each type.
            Calculate the Multiplier (Hurricane Avg Damage / Tropical Storm Avg Damage).

Question 5: [Risk Modeling with Smoothing] 
            Calculate the 'Risk Score' for each event type using a Damped Mean approach to handle sample imbalance.
            Formula: Risk Score = (Total Casualties + alpha * Global Mean) / (Count + alpha).
            - Casualties = Deaths Direct + Injuries Direct
            - Global Mean = Total Casualties (All Events) / Total Count (All Events)
            - alpha (Smoothing Factor) = 10
            Identify the Event Type with the highest Risk Score.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd
import numpy as np

try:
    storm_data = pd.read_csv("benchmarks/data_analysis/Meteorological/Florida_Storm/storm_data_florida_07012024_07312025.csv")
    drop_cols = [
    'STATE_ABBR', 'CZ_TIMEZONE', 'CZ_TYPE', 'CZ_FIPS', 'WFO', 
    'ABSOLUTE_ROWNUMBER', 'EPISODE_ID', 'EPISODE_NARRATIVE'
    ]
    storm_data.drop(columns=drop_cols, inplace=True, errors='ignore')
    storm_data = storm_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    storm_data = storm_data.replace('', np.nan)
except Exception as e:
    print(f"Error loading datasets: {e}")
    storm_data = None

correct_answers = {
    "q1_county": "BREVARD CO.",
    "q1_avg_damage": 2250.0,
    "q2_duration": 135.0,
    "q3_pct": 7.35,
    "q4_ts_avg": 14214.88,
    "q4_hur_avg": 251338823.53,
    "q4_multiplier": 17681.39,
    "q5_type": "Lightning",
    "q5_score": 0.374
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation for Glass Jaw County
    glass_jaw_county = runtime.get_variable("glass_jaw_county")
    glass_jaw_avg_damage = runtime.get_variable("glass_jaw_avg_damage")
    
    # Check types
    if not isinstance(glass_jaw_county, str):
        return ValidatorResult(success=False, message=f"Q1: County name must be a string. Got {type(glass_jaw_county).__name__}.")
    if not isinstance(glass_jaw_avg_damage, (int, float)):
        return ValidatorResult(success=False, message=f"Q1: Average damage must be a number. Got {type(glass_jaw_avg_damage).__name__}.")
    
    # Check values with tolerance for float
    county_match = glass_jaw_county.strip().upper() == correct_answers["q1_county"].strip().upper()
    damage_match = round(glass_jaw_avg_damage, 1) == correct_answers["q1_avg_damage"]
    
    if county_match and damage_match:
        return ValidatorResult(success=True, message=f"Q1 Correct: Identified the most fragile county for low-magnitude storms. County: {glass_jaw_county}, Average Damage: {glass_jaw_avg_damage:.1f}")
    
    return ValidatorResult(success=False, message=f"Q1 Incorrect. Expected {correct_answers['q1_county']} (${correct_answers['q1_avg_damage']:.2f}), but got {glass_jaw_county} (${glass_jaw_avg_damage:.1f})")

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    avg_flood_duration = runtime.get_variable("avg_flood_duration")

    if not isinstance(avg_flood_duration, (int, float)):
        return ValidatorResult(success=False, message=f"Q2: Average duration must be a number. Got {type(avg_flood_duration).__name__}.")

    if round(avg_flood_duration, 1) == correct_answers["q2_duration"]:
        return ValidatorResult(success=True, message=f"Q2 Correct: Average duration of Flash Floods that caused property damage > 0 is {avg_flood_duration:.1f} minutes")
    return ValidatorResult(success=False, message=f"Q2 Incorrect. Expected {correct_answers['q2_duration']}, got {avg_flood_duration:.1f}")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validate Pareto Principle
    top_damage_event_pct = runtime.get_variable("top_damage_event_pct")

    if not isinstance(top_damage_event_pct, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q3: Percentage must be a float. Got {type(top_damage_event_pct).__name__}")
    
    # Allow for some tolerance in percentage calculation
    if abs(top_damage_event_pct - correct_answers["q3_pct"]) < 0.5:
        return ValidatorResult(success=True, message="Q3 Correct")
    
    return ValidatorResult(success=False, message=f"Q3 Incorrect. Expected {correct_answers['q3_pct']}%, got {top_damage_event_pct}%")

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validate Intensity Analysis
    avg_damage_ts = runtime.get_variable("avg_damage_ts")
    avg_damage_hurricane = runtime.get_variable("avg_damage_hurricane")
    damage_multiplier = runtime.get_variable("damage_multiplier")

    if not isinstance(avg_damage_ts, (float, np.floating)) or not isinstance(avg_damage_hurricane, (float, np.floating)) or not isinstance(damage_multiplier, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q4: Averages and multiplier must be floats. Got {type(avg_damage_ts).__name__}, {type(avg_damage_hurricane).__name__}, {type(damage_multiplier).__name__}.")
    
    ts_correct = round(avg_damage_ts, 2) == correct_answers["q4_ts_avg"]
    hur_correct = round(avg_damage_hurricane, 2) == correct_answers["q4_hur_avg"]
    mult_correct = round(damage_multiplier, 2) == correct_answers["q4_multiplier"]

    if ts_correct and hur_correct and mult_correct:
        return ValidatorResult(success=True, message="Q4 Correct")
    
    return ValidatorResult(success=False, message=f"Q4 Incorrect. Expected TS: {correct_answers['q4_ts_avg']}, Hur: {correct_answers['q4_hur_avg']}, Mult: {correct_answers['q4_multiplier']}. Got {avg_damage_ts:.2f}, {avg_damage_hurricane:.2f}, {damage_multiplier:.2f}")

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    deadliest_event_type = runtime.get_variable("deadliest_event_type")
    risk_score = runtime.get_variable("risk_score")
    
    # Validation logic for Risk Score
    if not isinstance(deadliest_event_type, str):
         return ValidatorResult(success=False, message=f"Q5: Event type must be string. Got {type(deadliest_event_type).__name__}")
    if not isinstance(risk_score, (float, np.floating)):
         return ValidatorResult(success=False, message=f"Q5: Risk score must be float. Got {type(risk_score).__name__}")

    # Check type name
    type_match = deadliest_event_type.strip().lower() == correct_answers["q5_type"].strip().lower()
    # Check score with tolerance (0.01)
    score_match = abs(risk_score - correct_answers["q5_score"]) < 0.01

    if type_match and score_match:
        return ValidatorResult(success=True, message="Q5 Correct")
    return ValidatorResult(success=False, message=f"Q5 Incorrect. Expected {correct_answers['q5_type']} with score {correct_answers['q5_score']:.3f}, but got {deadliest_event_type} with score {risk_score:.3f}")

variables = [
    Variable(
        name="storm_data",
        value=storm_data,
        description="""DataFrame containing NOAA Storm Data in Florida during the period of 2024/07/01-2025/07/31.
        Columns and their dtypes:
        - EVENT_ID: int64 (Unique identifier for each event)
        - CZ_NAME_STR: object (string, County Name)
        - BEGIN_LOCATION: object (string, Start location of the event)
        - BEGIN_DATE: object (string, format: "MM/DD/YYYY" e.g., "07/01/2024")
        - BEGIN_TIME: int64 (Time in HHMM format, e.g., 1551 for 15:51)
        - EVENT_TYPE: object (string, Type of storm, e.g., "Thunderstorm Wind", "Heavy Rain")
        - MAGNITUDE: object (string, Physical intensity)
        - TOR_F_SCALE: object (string, Tornado scale)
        - DEATHS_DIRECT: int64 (Direct deaths caused by the event)
        - INJURIES_DIRECT: int64 (Direct injuries caused by the event)
        - DAMAGE_PROPERTY_NUM: int64 (Property damage amount)
        - DAMAGE_CROPS_NUM: int64 (Crops damage amount)
        - MAGNITUDE_TYPE: object (string, Type of magnitude measurement, e.g., "MG", may be NaN)
        - INJURIES_INDIRECT: int64 (Indirect injuries)
        - DEATHS_INDIRECT: int64 (Indirect deaths)
        - SOURCE: object (string, Data source, e.g., "ASOS", "Mesonet", "Public")
        - FLOOD_CAUSE: object (string, Flood cause, may be NaN)
        - TOR_LENGTH: object (string, Tornado length, may be NaN)
        - TOR_WIDTH: object (string, Tornado width, may be NaN)
        - BEGIN_RANGE: object (string, Begin range, may be numeric string like "6" or NaN)
        - BEGIN_AZIMUTH: object (string, Begin azimuth, e.g., "NE", "SW", may be NaN)
        - END_RANGE: object (string, End range, may be numeric string or NaN)
        - END_AZIMUTH: object (string, End azimuth, e.g., "NE", "SW", may be NaN)
        - END_LOCATION: object (string, End location of the event)
        - END_DATE: object (string, format: "MM/DD/YYYY")
        - END_TIME: int64 (Time in HHMM format)
        - BEGIN_LAT: object (string, Latitude as string, e.g., "30.36")
        - BEGIN_LON: object (string, Longitude as string, e.g., "-85.8")
        - END_LAT: object (string, Latitude as string)
        - END_LON: object (string, Longitude as string)
        - EVENT_NARRATIVE: object (string, Text description of the event, may be NaN)
        Example: pd.DataFrame(data={
            "EVENT_ID": [1191836],
            "CZ_NAME_STR": ["BAY CO."],
            "BEGIN_LOCATION": ["WESTBAY"],
            "BEGIN_DATE": ["07/01/2024"],
            "BEGIN_TIME": [1551],
            "EVENT_TYPE": ["Thunderstorm Wind"],
            "MAGNITUDE": ["57.00"],
            "TOR_F_SCALE": [None],
            "DEATHS_DIRECT": [0],
            "INJURIES_DIRECT": [0],
            "DAMAGE_PROPERTY_NUM": [0],
            "DAMAGE_CROPS_NUM": [0],
            "MAGNITUDE_TYPE": ["MG"],
            "INJURIES_INDIRECT": [0],
            "DEATHS_INDIRECT": [0],
            "SOURCE": ["ASOS"],
            "FLOOD_CAUSE": [None],
            "TOR_LENGTH": [None],
            "TOR_WIDTH": [None],
            "BEGIN_RANGE": ["6"],
            "BEGIN_AZIMUTH": ["NE"],
            "END_RANGE": ["6"],
            "END_AZIMUTH": ["NE"],
            "END_LOCATION": ["WESTBAY"],
            "END_DATE": ["07/01/2024"],
            "END_TIME": [1551],
            "BEGIN_LAT": ["30.36"],
            "BEGIN_LON": ["-85.8"],
            "END_LAT": ["30.36"],
            "END_LON": ["-85.8"],
            "EVENT_NARRATIVE": [None],
        })
        """
    ),
    # Q1 Variables
    Variable(
        name="glass_jaw_county", 
        value="", 
        description="String. The name of the county with the highest average damage from low-magnitude storms (<50 knots) of question 1."
    ),
    Variable(
        name="glass_jaw_avg_damage", 
        value=0.0, 
        description="Float. The calculated average property damage for that county of question 1."
    ),
    
    # Q2 Variables
    Variable(
        name="avg_flood_duration", 
        value=0.0, 
        description="Float. Average duration in minutes for damaging Flash Floods that caused property damage > 0 of question 2."),
    
    # Q3 Variables
    Variable(
        name="top_damage_event_pct", 
        value=0.0, 
        description="Float. The percentage of top damaging events (0-100) required to account for 80% of total damage of question 3. Round up to the next integer number of events if the 80% threshold falls between events, then calculate percentage as (number of events / total events with damage > 0) * 100."
    ),
    
    # Q4 Variables
    Variable(
        name="avg_damage_ts", 
        value=0.0, 
        description="Float. The average property damage for 'Tropical Storm' events of question 4."
    ),
    Variable(
        name="avg_damage_hurricane", 
        value=0.0, 
        description="Float. The average property damage for 'Hurricane (Typhoon)' events of question 4."
    ),
    Variable(
        name="damage_multiplier", 
        value=0.0, 
        description="Float. The ratio of Hurricane damage to Tropical Storm damage of question 4."
    ),
    
    # Q5 Variables
    Variable(
        name="deadliest_event_type", 
        value="", 
        description="String. The event type with the highest Risk Score of question 5."
    ),
    Variable(
        name="risk_score", 
        value=0.0, 
        description="Float. The calculated Risk Score for that event type of question 5."
    )
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