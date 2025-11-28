"""
This scenario is based on analyzing traffic data from Hong Kong road segments.
The goal is to use pandas and data analysis techniques to progressively analyze
the relationship between road segment characteristics and traffic conditions.

Table Structure:
1. road_segment:
   - id (primary key), street_id (foreign key), road_segment_code (varchar)
   - length (smallint), road_level (smallint), direction (enum)
   - geometry (geometry), free_flow_speed (real), created_at (timestamp)

2. hourly_road_segment_traffic:
   - id (primary key), road_segment_id (foreign key to road_segment.id)
   - speed (real), congestion_status (smallint), congestion_index (real)
   - datetime (timestamp)
   - unique constraint on (road_segment_id, datetime)

Direction enum values: 'FORWARD_ONLY', 'BIDIRECTIONAL'
Congestion status: 0 (Free flow), 1 (Light congestion), 2 (Heavy congestion), 3 (Severe congestion)
Road level: -1 (Tunnel), 0 (Ground level), 1/2 (Elevated/Bridge)

Question 1: Count road segments with FORWARD_ONLY direction (Single Table Query)
Please count the number of road segments where the `direction` is 'FORWARD_ONLY' from the `road_segment.csv` file.

Question 2: Count unique road segments with heavy congestion on 2025-07-27 (Single Table Query)
Using the `hourly_road_segment_traffic.csv` file, please count how many unique road segments had `congestion_status` equal to 2 (heavy congestion) on the date '2025-07-27'.

Question 3: Find average speed for elevated roads during peak hours from 2025-08-01 to 2025-08-04 (Multi-turn 1/3, Join Required)
By joining `hourly_road_segment_traffic.csv` and `road_segment.csv`, calculate the average speed for road segments where `road_level` is 1/2 (elevated roads/bridges) during peak hours (08:00-10:00 and 17:00-19:00) from '2025-08-01' to '2025-08-04'.

Question 4: Calculate average congestion index for ground level roads during weekdays (Multi-turn 2/3)
Based on ground level roads (road_level = 0), calculate the average congestion_index for weekdays (Monday to Friday) during the period from '2025-08-01' to '2025-08-04'.

Question 5: Compare congestion patterns between elevated and tunnel roads (Multi-turn 3/3)
Calculate the average congestion_index for elevated roads (road_level = 1/2) and tunnel roads (road_level = -1) during the period '2025-08-01' to '2025-08-04', then determine which road level has higher average congestion.

"""

from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn
import pandas as pd
from typing import List
from core.types import ToolCall

# Load the data
road_segment_df = pd.read_csv("benchmarks/data_analysis/HKTraffic/road_segment.csv")
hourly_road_segment_traffic_df = pd.read_csv("benchmarks/data_analysis/HKTraffic/hourly_road_segment_traffic.csv")

tools = []

# Correct answers (these would be calculated from actual data)
correct_answers = {
    "q1": 4084,  # Number of forward_only road segments
    "q2": 870,   # Number of unique road segments with heavy congestion on 2025-07-27
    "q3": 65, # Average speed for elevated roads during peak hours
    "q4": 1.2, # Average congestion index for ground level roads during weekdays
    "q5_elevated_congestion": 1.14,    # Average congestion index for elevated roads
    "q5_tunnel_congestion": 1.15,     # Average congestion index for tunnel roads
    "q5_comparison": "Tunnel"       # Which road level has higher congestion
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the count of forward_only road segments.
    """
    forward_only_count = runtime.get_variable("forward_only_count")
    expected = correct_answers["q1"]
    
    if forward_only_count is not None and forward_only_count == expected:
        return ValidatorResult(
            success=True,
            message=f"Correct. There are {forward_only_count} road segments with forward_only direction."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect count. Expected {expected}, but got {forward_only_count}."
        )

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the count of road segments with heavy congestion on 2025-07-27.
    """
    congestion_count_2025_07_27 = runtime.get_variable("congestion_count_2025_07_27")
    expected = correct_answers["q2"]
    
    if congestion_count_2025_07_27 is not None and congestion_count_2025_07_27 == expected:
        return ValidatorResult(
            success=True,
            message=f"Correct. There were {congestion_count_2025_07_27} road segments with heavy congestion on 2025-07-27."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect count. Expected {expected}, but got {congestion_count_2025_07_27}."
        )

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the average speed for elevated roads during peak hours.
    """
    avg_speed_elevated_peak_hours = runtime.get_variable("avg_speed_elevated_peak_hours")
    expected = correct_answers["q3"]
    
    if avg_speed_elevated_peak_hours is not None and round(avg_speed_elevated_peak_hours) == expected:
        return ValidatorResult(
            success=True,
            message=f"Correct. The average speed for elevated roads during peak hours is {avg_speed_elevated_peak_hours} km/h."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect average speed. Expected around {expected}, but got {avg_speed_elevated_peak_hours}."
        )

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the average congestion index for ground level roads during weekdays.
    """
    avg_congestion_ground_weekdays = runtime.get_variable("avg_congestion_ground_weekdays")
    expected = correct_answers["q4"]
    
    if avg_congestion_ground_weekdays is not None and round(avg_congestion_ground_weekdays,1) == expected:
        return ValidatorResult(
            success=True,
            message=f"Correct. The average congestion index for ground level roads during weekdays is {avg_congestion_ground_weekdays:.1f}."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect average congestion index. Expected around {expected}, but got {avg_congestion_ground_weekdays:.1f}."
        )
def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the congestion comparison between elevated and tunnel roads.
    """
    avg_congestion_elevated = runtime.get_variable("avg_congestion_elevated")
    avg_congestion_tunnel = runtime.get_variable("avg_congestion_tunnel")
    congestion_comparison_result = runtime.get_variable("congestion_comparison_result")
    
    expected_elevated = correct_answers["q5_elevated_congestion"]
    expected_tunnel = correct_answers["q5_tunnel_congestion"]
    expected_comparison = correct_answers["q5_comparison"]
    
    elevated_correct = avg_congestion_elevated is not None and round(avg_congestion_elevated,2) == expected_elevated
    tunnel_correct = avg_congestion_tunnel is not None and round(avg_congestion_tunnel,2) == expected_tunnel
    comparison_correct = congestion_comparison_result is not None and congestion_comparison_result.strip().lower() == expected_comparison.strip().lower()
    
    if elevated_correct and tunnel_correct and comparison_correct:
        return ValidatorResult(
            success=True,
            message=f"Correct. Elevated roads average congestion: {avg_congestion_elevated:.2f}, Tunnel roads: {avg_congestion_tunnel:.2f}. Higher congestion: {congestion_comparison_result}."
        )
    else:
        error_parts = []
        if not elevated_correct:
            error_parts.append(f"Elevated congestion expected {expected_elevated}, got {avg_congestion_elevated}")
        if not tunnel_correct:
            error_parts.append(f"Tunnel congestion expected {expected_tunnel}, got {avg_congestion_tunnel}")
        if not comparison_correct:
            error_parts.append(f"Comparison expected {expected_comparison}, got {congestion_comparison_result}")
        
        return ValidatorResult(
            success=False,
            message=f"Incorrect result. {'; '.join(error_parts)}."
        )

variables = [
    Variable(
        name="road_segment_df",
        value=road_segment_df,
        description="""A pandas DataFrame from 'road_segment.csv' containing information for road segments.
        Columns and their dtypes:
        - id: int64
        - street_id: int64
        - road_segment_code: object (string)
        - length: int64
        - road_level: int64
        - direction: object (string)
        - geometry: object (string)
        - free_flow_speed: float64
        - created_at: object (string)
        example: pd.DataFrame(
            {
                "id": [1],
                "street_id": [101],
                "road_segment_code": ["593"],
                "length": [500],
                "road_level": [1],
                "direction": ["forward_only"],
                "geometry": ["0105000020E61000000100000001020000000300000012F41B66618B5C40021266BA6C463640695F24CA608B5C400DEBA1756C4636407B0E01725E8B5C403633016D6B463640"],
                "free_flow_speed": [60.0],
                "created_at": ["2025-01-01 00:00:00"],
            }
        )
        You should get the road_segment_df from this variable to answer the question.
        """
    ),
    Variable(
        name="hourly_road_segment_traffic_df",
        value=hourly_road_segment_traffic_df,
        description="""A pandas DataFrame from 'hourly_road_segment_traffic.csv' containing hourly traffic readings.
        Columns and their dtypes:
        - id: int64
        - road_segment_id: int64
        - speed: float64
        - congestion_status: int64
        - congestion_index: float64
        - datetime: object (string)
        example: pd.DataFrame(
            {
                "id": [1001],
                "road_segment_id": [1],
                "speed": [45.5],
                "congestion_status": [1],
                "congestion_index": [1.3],
                "datetime": ["2025-08-01 03:00:00.000000"],
            }
        )
        You should get the hourly_road_segment_traffic_df from this variable to answer the question.
        """
    ),
    Variable(
        name="forward_only_count",
        value=0,
        description="Stored the count of road segments with forward_only direction."
    ),
    Variable(
        name="congestion_count_2025_07_27",
        value=0,
        description="Stored the count of road segments with congestion_status equals to 2 on 2025-07-27."
    ),
    Variable(
        name="avg_speed_elevated_peak_hours",
        value=0.0,
        description="Stored the average speed for roads with road_level equals to 1 or 2 during peak hours (08:00-10:00 and 17:00-19:00) from 2025-08-01 to 2025-08-04."
    ),
    Variable(
        name="avg_congestion_ground_weekdays",
        value=0.0,
        description="Stored the average congestion index for roads with road_level equals to 0 during weekdays from 2025-08-01 to 2025-08-04."
    ),
    Variable(
        name="avg_congestion_elevated",
        value=0.0,
        description="Stored the average congestion index for roads with road_level equals to 1 or 2 (elevated) from 2025-08-01 to 2025-08-04."
    ),
    Variable(
        name="avg_congestion_tunnel",
        value=0.0,
        description="Stored the average congestion index for roads with road_level equals to -1 (tunnel) from 2025-08-01 to 2025-08-04."
    ),
    Variable(
        name="congestion_comparison_result",
        value="",
        description="Stored the result of comparison between roads with road_level equals to 1 or 2 (elevated) and roads with road_level equals to -1 (tunnel), example: 'elevated' or 'tunnel'."
    )
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