"""
This scenario is based on analyzing NYC Yellow Taxi Trip Data (2025-01 Sample).
The goal is to use raw trip records to perform spatial-temporal analysis and policy impact checks.

Dataset Structure:
taxi_df contains:
- tpep_pickup_datetime, tpep_dropoff_datetime: Strings (need parsing)
- PULocationID, DOLocationID: Integer (Zone IDs)
- trip_distance: Float (Miles)
- fare_amount, tip_amount, total_amount: Float (USD)
- congestion_surcharge: Float (USD)
- [Other columns]: VendorID, passenger_count, payment_type, etc.

Question 1: Congestion Analysis. Calculate average speed (MPH) by zone and find the slowest zone.
Question 2: Spatial Tidal Flow. Find the zone with highest Net Outflow (Pickups - Dropoffs) during morning rush.
Question 3: Economic Impact. Compare Surcharge Burden (Surcharge/Fare) for Short (<2mi) vs Long (>10mi) trips.
Question 4: Network Complexity. Find the PULocationID with the highest Shannon Entropy of destination distribution.
Question 5: Operational Efficiency. Find the 'Sink Zone' (Highest Dropoff/Pickup Imbalance) to identify deadhead risks.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd
import numpy as np

# Load the dataset 
try:
    taxi_df = pd.read_csv("benchmarks/data_analysis/Geospatial/NYC_TCL/yellow_tripdata_2025_01.csv")
    # Pre-processing to ensure dates are datetime objects, as raw CSV is string
    taxi_df['tpep_pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'])
    taxi_df['tpep_dropoff_datetime'] = pd.to_datetime(taxi_df['tpep_dropoff_datetime'])
except Exception as e:
    print(f"Error loading Taxi dataset: {e}")
    taxi_df = None

tools = []

# Correct answers
correct_answers = {
    "q1_id": 187,           # Midtown Center (Likely very slow)
    "q1_speed": 9.14,        # Very low speed (mph) in congested areas
    "q2": 186,              # Net Outflow for the zone with highest positive Net Outflow during morning rush
    "q3_short": 0.2827,         # Short trips surcharge burden
    "q3_long": 0.0265,         # Long trips surcharge burden
    "q4_id": 132,           # Midtown Center
    "q4_entropy": 6.9669,     # High entropy value (log(e) base usually)
    "q5_id": 1,            # Sink Zone id with highest imbalance ratio
    "q5_ratio": 0.8960        # Imbalance ratio for the zone with highest imbalance ratio
}

# --- Validators ---

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    slowest_zone_id = runtime.get_variable("slowest_zone_id")
    slowest_zone_speed = runtime.get_variable("slowest_zone_speed")

    # Check type
    if not isinstance(slowest_zone_id, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q1: Expected int (ID), got {type(slowest_zone_id).__name__}")
    if not isinstance(slowest_zone_speed, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q1: Expected float (Speed), got {type(slowest_zone_speed).__name__}")
    
    val = round(slowest_zone_speed, 2)
    id_correct = slowest_zone_id == correct_answers["q1_id"]
    speed_correct = val == correct_answers["q1_speed"]

    if id_correct and speed_correct:
        return ValidatorResult(success=True, message="Q1: Correct.")
    return ValidatorResult(success=False, message=f"Q1: Incorrect. Expected ID {correct_answers['q1_id']} (Speed ~{correct_answers['q1_speed']}), got ID {slowest_zone_id} (Speed {val})")

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    top_source_zone_id = runtime.get_variable("top_source_zone_id")

    # Check type
    if not isinstance(top_source_zone_id, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q2: Expected int (Zone ID), got {type(top_source_zone_id).__name__}")
    
    # Check value
    if top_source_zone_id == correct_answers["q2"]:
        return ValidatorResult(success=True, message="Q2: Correct.")
    return ValidatorResult(success=False, message=f"Q2: Incorrect. Expected Zone ID {correct_answers['q2']}, got {top_source_zone_id}")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    surcharge_burden = runtime.get_variable("surcharge_burden_by_distance")

    if not isinstance(surcharge_burden, pd.Series):
        return ValidatorResult(success=False, message=f"Q3: Expected pd.Series, got {type(surcharge_burden).__name__}")
    
    try:
        short_val = round(surcharge_burden.get('Short Trips', 0), 4)
        long_val = round(surcharge_burden.get('Long Trips', 0), 4)
    except:
        return ValidatorResult(success=False, message="Q3: Series index must contain 'Short Trips' and 'Long Trips'.")
    
    short_correct = short_val == correct_answers["q3_short"]
    long_correct = long_val == correct_answers["q3_long"]
    # Check values with some tolerance
    if short_correct and long_correct:
        return ValidatorResult(success=True, message="Q3: Correct.")
    return ValidatorResult(success=False, message=f"Q3: Incorrect. Expected Short~{correct_answers['q3_short']}, Long~{correct_answers['q3_long']}. Got Short={short_val}, Long={long_val}")

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    top_entropy_zone_id = runtime.get_variable("top_entropy_zone_id")
    top_entropy_value = runtime.get_variable("top_entropy_value")

    # Check type
    if not isinstance(top_entropy_zone_id, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q4: Expected int for Zone ID, got {type(top_entropy_zone_id).__name__}")
    if not isinstance(top_entropy_value, (float, np.number)):
        return ValidatorResult(success=False, message=f"Q4: Expected float for Entropy, got {type(top_entropy_value).__name__}")

    val = round(float(top_entropy_value), 4)

    # Check values
    id_correct = top_entropy_zone_id == correct_answers["q4_id"]
    val_correct = val == correct_answers["q4_entropy"] # Allow slight variance in entropy calc

    if id_correct and val_correct:
        return ValidatorResult(success=True, message="Q4: Correct.")
    return ValidatorResult(success=False, message=f"Q4: Incorrect. Expected ID {correct_answers['q4_id']} (Entropy ~{correct_answers['q4_entropy']}), got ID {top_entropy_zone_id} (Entropy {val})")

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    top_sink_zone_id = runtime.get_variable("top_sink_zone_id")
    sink_imbalance_ratio = runtime.get_variable("sink_imbalance_ratio")

    # Check type
    if not isinstance(top_sink_zone_id, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q5: Expected int for Zone ID, got {type(top_sink_zone_id).__name__}")
    if not isinstance(sink_imbalance_ratio, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q5: Expected float for Ratio, got {type(sink_imbalance_ratio).__name__}")

    val = round(sink_imbalance_ratio, 4)
    
    # Check values
    id_correct = top_sink_zone_id == correct_answers["q5_id"]
    # Ratio might vary slightly based on data subset, allow tolerance
    ratio_correct = val == correct_answers["q5_ratio"]

    if id_correct and ratio_correct:
        return ValidatorResult(success=True, message="Q5: Correct.")
    return ValidatorResult(success=False, message=f"Q5: Incorrect. Expected ID {correct_answers['q5_id']} (Ratio ~{correct_answers['q5_ratio']}), got ID {top_sink_zone_id} (Ratio {val})")


variables = [
    # Dataset
    Variable(
        name="taxi_df",
        value=taxi_df,
        description="""
        A pandas DataFrame containing 2025 January NYC Yellow Taxi trip records.
        Note: The datetime columns (tpep_pickup_datetime and tpep_dropoff_datetime) have been pre-processed and are already datetime64[ns] objects, ready to use without additional parsing.
        Columns and their dtypes:
        - VendorID: int64
        - tpep_pickup_datetime: datetime64[ns] 
        - tpep_dropoff_datetime: datetime64[ns]
        - passenger_count: float64
        - trip_distance: float64 (miles)
        - RatecodeID: float64
        - store_and_fwd_flag: object (string, 'N' or 'Y')
        - PULocationID: int64 (Pickup Zone ID)
        - DOLocationID: int64 (Dropoff Zone ID)
        - payment_type: int64
        - fare_amount: float64 (USD)
        - extra: float64 (USD)
        - mta_tax: float64 (USD)
        - tip_amount: float64 (USD)
        - tolls_amount: float64 (USD)
        - improvement_surcharge: float64 (USD)
        - total_amount: float64 (USD)
        - congestion_surcharge: float64 (USD, typically 2.5 for Manhattan south of 96th St, 0.0 otherwise)
        - Airport_fee: float64 (USD)
        - cbd_congestion_fee: float64 (USD)
        Example: pd.DataFrame(data={
            "VendorID": [1],
            "tpep_pickup_datetime": [pd.Timestamp("2025-01-01 00:18:38")],  
            "tpep_dropoff_datetime": [pd.Timestamp("2025-01-01 00:26:59")], 
            "passenger_count": [1.0],
            "trip_distance": [1.60],
            "RatecodeID": [1.0],
            "store_and_fwd_flag": ["N"],
            "PULocationID": [229],
            "DOLocationID": [237],
            "payment_type": [1],
            "fare_amount": [10.0],
            "extra": [3.5],
            "mta_tax": [0.5],
            "tip_amount": [3.00],
            "tolls_amount": [0.0],
            "improvement_surcharge": [1.0],
            "total_amount": [18.00],
            "congestion_surcharge": [2.5],
            "Airport_fee": [0.0],
            "cbd_congestion_fee": [0.0],
        })
        Use this dataframe for all queries.
        """
    ),
    # Q1
    Variable(
        name="slowest_zone_id",
        value=0,
        description="Int. The PULocationID with the lowest average speed.",
    ),
    Variable(
        name="slowest_zone_speed",
        value=0.0,
        description="Float. The average speed (mph) of the slowest zone.",
    ),
    # Q2
    Variable(
        name="top_source_zone_id",
        value=0,
        description="Int. The Zone ID with the highest Net Outflow during morning rush.",
    ),
    # Q3
    Variable(
        name="surcharge_burden_by_distance",
        value=pd.Series(dtype=float),
        description="Series. The Surcharge Burden for Short (<2mi) and Long (>10mi) trips indexed by ['Short Trips', 'Long Trips'].",
    ),
    # Q4
    Variable(
        name="top_entropy_zone_id",
        value=0,
        description="Int. The PULocationID with the highest Shannon Entropy of its destination distribution.",
    ),
    Variable(
        name="top_entropy_value",
        value=0.0,
        description="Float. The calculated Shannon Entropy value for the top zone.",
    ),
    # Q5
    Variable(
        name="top_sink_zone_id",
        value=0,
        description="Int. The Zone ID with the highest Imbalance Ratio ((DO-PU)/(DO+PU)).",
    ),
    Variable(
        name="sink_imbalance_ratio",
        value=0.0,
        description="Float. The imbalance ratio for the top sink zone.",
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