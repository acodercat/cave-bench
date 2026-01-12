"""
This scenario is based on analyzing air quality data from Hong Kong monitoring stations.
The goal is to use pandas and data analysis techniques to progressively analyze
the relationship between station characteristics and air quality measurements.

Table Structure:
1. air_quality_station:
   - id (primary key), name_en, name_tc, address_en, address_tc
   - coordinate (geography point), altitude (double precision)
   - station_type (enum), area_type (enum), created_at (timestamp)

2. hourly_station_air_quality:
   - id (primary key), station_id (foreign key to air_quality_station.id)
   - aqhi (integer), pm2_5, pm10, o3, so2, no2, co (double precision)
   - no2_ar, o3_ar, pm_ar, so2_ar (double precision), pm_ar_source (enum)
   - datetime (timestamp), aqhi_level (enum)
   - unique constraint on (station_id, datetime)


Question 1: Calculate average altitude for NEW_TOWN stations (Single Table Query)
Please calculate the average altitude (`altitude`) for all stations where the `area_type` is 'NEW_TOWN' from the `air_quality_station.csv` file.

Question 2: Find the highest NO2 concentration for station_id 5 on 2025-07-25 (Single Table Query)
Using the `hourly_station_air_quality.csv` file, please find the highest nitrogen dioxide (`no2`) concentration recorded for the station with `station_id` 5 on the date '2025-07-25'.

Question 3: Find the station with the highest PM10 reading (Multi-turn 1/3, Join Required)
By joining `hourly_station_air_quality.csv` and `air_quality_station.csv`, find the single highest recorded `pm10` value. Please return the English name (`name_en`) of the station where this occurred, its area type (`area_type`), the recorded `datetime`, and the `pm10` value itself.

Question 4: Calculate average PM2.5 for the station found in Q3 on 2025-07-26 (Multi-turn 2/3)
Based on the station identified in the previous step ('Southern'), calculate its average `pm2_5` concentration for the date '2025-07-26' only.

Question 5: Calculate average PM2.5 for Central on 2025-07-26 and compare with Southern (Multi-turn 3/3)
Calculate the average `pm2_5` concentration for all stations in the Central area on the date '2025-07-26', then compare it with the Southern station's average PM2.5 (from Q4) to determine which one has a higher value.

"""

from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn
import pandas as pd
from typing import List
from core.types import ToolCall

hourly_station_air_quality_df = pd.read_csv("benchmarks/data_analysis/HKAirQuality/hourly_station_air_quality.csv")
air_quality_station_df = pd.read_csv("benchmarks/data_analysis/HKAirQuality/air_quality_station.csv")
target_area_type_q1 = "NEW_TOWN"
target_station_id_q2 = 5
target_date_q2 = "2025-07-25"
target_date_q4_q5 = "2025-07-26"
target_name_en_q5 = "Central"

tools = []

correct_answers = {
    "q1": 25.92,
    "q2": 69.7,
    "q3": "Southern",
    "q4": 16.6,
    "q5_central_avg": 23.2,
    "q5_comparison": "Central"
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the calculation of average altitude for NEW_TOWN stations.
    """
    avg_altitude_new_town = runtime.retrieve("avg_altitude_new_town")
    expected = correct_answers["q1"]
    
    if avg_altitude_new_town is not None and round(avg_altitude_new_town, 2) == expected:
        return ValidatorResult(
            success=True,
            message=f"Correct. The average altitude for NEW_TOWN stations is {avg_altitude_new_town:.2f} meters."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect average altitude. Expected around {expected}, but got {avg_altitude_new_town}."
        )

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the highest NO2 concentration for station_id 5 on 2025-07-25.
    """
    max_no2_station5_2025_07_25 = runtime.retrieve("max_no2_station5_2025_07_25")
    expected = correct_answers["q2"]
    
    if max_no2_station5_2025_07_25 is not None and max_no2_station5_2025_07_25 == expected:
        return ValidatorResult(
            success=True,
            message=f"Correct. The highest NO2 concentration for station_id 5 on 2025-07-25 was {max_no2_station5_2025_07_25}."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect highest NO2 concentration for station_id 5. Expected {expected}, but got {max_no2_station5_2025_07_25}."
        )

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates finding the station with the highest single PM10 reading.
    """
    max_pm10_station_name = runtime.retrieve("max_pm10_station_name")
    expected = correct_answers["q3"]

    if max_pm10_station_name is not None and max_pm10_station_name.strip().lower() == expected.strip().lower():
        return ValidatorResult(
            success=True,
            message=f"Correct. The highest PM10 reading was recorded at the '{max_pm10_station_name}' station."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect result. Expected station '{expected}', but got '{max_pm10_station_name}'."
        )

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the average PM2.5 calculation for the 'Southern' station on 2025-07-26.
    """
    southern_avg_pm25_2025_07_26 = runtime.retrieve("southern_avg_pm25_2025_07_26")
    expected = correct_answers["q4"]
    
    if southern_avg_pm25_2025_07_26 is not None and round(southern_avg_pm25_2025_07_26, 1) == expected:
        return ValidatorResult(
            success=True,
            message=f"Correct. The average PM2.5 for the 'Southern' station on 2025-07-26 is {southern_avg_pm25_2025_07_26}."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect average PM2.5 value. Expected {expected}, but got {southern_avg_pm25_2025_07_26}."
        )

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the average PM2.5 for Central on 2025-07-26 and comparison with Southern.
    """
    central_avg_pm25_2025_07_26 = runtime.retrieve("central_avg_pm25_2025_07_26")
    comparison_result = runtime.retrieve("comparison_result")
    expected_central_avg = correct_answers["q5_central_avg"]
    expected_comparison = correct_answers["q5_comparison"]
    
    if central_avg_pm25_2025_07_26 is not None and round(central_avg_pm25_2025_07_26, 1) == expected_central_avg and comparison_result.strip().lower() == expected_comparison.strip().lower():
        return ValidatorResult(
            success=True,
            message=f"Correct. Central area average PM2.5 is {central_avg_pm25_2025_07_26}. Comparison result: {comparison_result}."
        )
    elif central_avg_pm25_2025_07_26 is not None and round(central_avg_pm25_2025_07_26, 1) != expected_central_avg and comparison_result.strip().lower() == expected_comparison.strip().lower():
        return ValidatorResult(
            success=False,
            message=f"Incorrect result. Expected Central average {expected_central_avg}, got {central_avg_pm25_2025_07_26}. Comparison result: {comparison_result}."
        )
    elif central_avg_pm25_2025_07_26 is not None and round(central_avg_pm25_2025_07_26, 1) == expected_central_avg and comparison_result.strip().lower() != expected_comparison.strip().lower():
        return ValidatorResult(
            success=False,
            message=f"Incorrect result. Central area average PM2.5 is {central_avg_pm25_2025_07_26}. But the comparison result is not the same as the expected result: {comparison_result}."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect result. Expected Central average {expected_central_avg}, got {central_avg_pm25_2025_07_26}. Comparison result is not the same as the expected result: {comparison_result}."
        )


variables = [
    Variable(
        name="air_quality_station_df",
        value=air_quality_station_df,
        description="""A pandas DataFrame from 'air_quality_station.csv'.
        Columns and their dtypes:
        - id: int64
        - name_en: object (string)
        - name_tc: object (string)
        - address_en: object (string)
        - address_tc: object (string)
        - coordinate: object (string)
        - altitude: float64
        - station_type: object (string)
        - area_type: object (string)
        - created_at: object (string)
        example: pd.DataFrame(
            {
                "id": [1],
                "name_en": ["Southern"],
                "name_tc": ["南區"],
                "address_en": ["No.1 Aberdeen Praya Road, Hong Kong."],
                "address_tc": ["香港香港仔海傍道1號"],
                "coordinate": ["0101000020E6100000CE1143BC3F8A5C4026704E99593F3640"],
                "altitude": [18],
                "station_type": ["GENERAL"],
                "created_at": ["2025-01-01 00:00:00"],
                "area_type": ["URBAN"],
            }
        )
        You should get the air_quality_station_df from this variable to answer the question.
        """
    ),
    Variable(
        name="hourly_station_air_quality_df",
        value=hourly_station_air_quality_df,
        description="""A pandas DataFrame from 'hourly_station_air_quality.csv'.
        Columns and their dtypes:
        - id: int64
        - station_id: int64
        - aqhi: int64
        - pm2_5: float64
        - pm10: float64
        - o3: float64
        - so2: float64
        - no2: float64
        - co: float64
        - no2_ar: float64
        - o3_ar: float64
        - pm_ar: float64
        - so2_ar: float64
        - pm_ar_source: object (string)
        - datetime: object (string)
        - aqhi_level: object (string)
        example: pd.DataFrame(
            {
                "id": [618675],
                "station_id": [1],
                "aqhi": [3],
                "pm2_5": [13.6],
                "pm10": [92.7],
                "o3": [56.1],
                "so2": [12.6],
                "no2": [53],
                "co": [449.4],
                "no2_ar": [2.361],
                "o3_ar": [2.53],
                "pm_ar": [0.509],
                "so2_ar": [0.16],
                "pm_ar_source": ["PM10"],
                "datetime": ["2025-01-01 00:00:00"],
                "aqhi_level": ["LOW"],
            }
        )
        You should get the hourly_station_air_quality_df from this variable to answer the question.
        """
    ),
    Variable(
        name='target_area_type_q1',
        value=target_area_type_q1,
        description="The area type of the target station for Q1, example: 'NEW_TOWN', you should get the area type from this variable to answer question 1."
    ),
    Variable(
        name='target_station_id_q2',
        value=target_station_id_q2,
        description="The station id of the target station for Q2, example: 5, you should get the station id from this variable to answer question 2."
    ),
    Variable(
        name='target_date_q2',
        value=target_date_q2,
        description="The date of the target station for Q2, example: '2025-07-25', you should get the date from this variable to answer question 2."
    ),
    Variable(
        name='target_date_q4_q5',
        value=target_date_q4_q5,
        description="The date of the target station for Q4 and Q5, example: '2025-07-26', you should get the date from this variable to answer question 4 and 5."
    ),
    Variable(
        name='target_name_en_q5',
        value=target_name_en_q5,
        description="The English name of the target station for Q5, example: 'Central', you should get the English name from this variable to answer question 5."
    ),
    Variable(
        name="avg_altitude_new_town",
        value=0.0,
        description="Stored the calculated average altitude of all 'NEW_TOWN' stations."
    ),
    Variable(
        name="max_no2_station5_2025_07_25",
        value=0.0,
        description="Stored the highest NO2 concentration recorded for station_id 5 on 2025-07-25."
    ),
    Variable(
        name="max_pm10_station_name",
        value="",
        description="Stored the English name of the station with the single highest PM10 reading."
    ),
    Variable(
        name="southern_avg_pm25_2025_07_26",
        value=0.0,
        description="Stored the calculated average PM2.5 concentration for the 'Southern' station on 2025-07-26."
    ),
    Variable(
        name="central_avg_pm25_2025_07_26",
        value=0.0,
        description="Stored the calculated average PM2.5 concentration for 'Central' station on 2025-07-26."
    ),
    Variable(
        name="comparison_result",
        value="",
        description="Stored the station name indicating which has higher PM2.5: 'Central' or 'Southern'."
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