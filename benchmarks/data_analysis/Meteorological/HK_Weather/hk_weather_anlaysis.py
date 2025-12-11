"""
This scenario is based on analyzing Hong Kong meteorological data for the year 2025.
The goal is to use pandas to perform advanced meteorological analysis combining 
observational data with geospatial station attributes.

Dataset Structure:
1. 'hk_weather': High-frequency weather observations.
   - Columns: id, station_id, temperature, visibility, pressure, humidity, wind_degree, wind_speed, wind_gust, datetime
2. 'hk_stations': Station metadata including location and altitude.
   - Columns: id, name_en, station_code, latitude, longitude, altitude, station_type

Question 1: [Cross-Border Pollution Analysis]
Analyze the "Dry Haze" phenomenon. Calculate the probability of a "Dry Haze Event" (Visibility < 8000m AND Humidity < 80%) 
occurring under Winter Monsoon winds (North/Northwest: 315°-360° or 0°-45°) versus Summer Monsoon winds (South/Southwest: 135°-225°).
Return the ratio: (Probability_Winter / Probability_Summer).

Question 2: [Urban Heat Island (UHI) Intensity]
Calculate the "Nighttime UHI Intensity". 
1. Filter for nighttime hours (20:00 to 05:59).
2. Identify the station with the highest average nighttime temperature (Urban proxy) and the station with the lowest (Rural proxy).
3. Calculate the difference between their average nighttime temperatures.

Question 3: [Urban Canyon Effect / Gust Factor]
Calculate the average "Gust Factor" (Wind Gust / Wind Speed) for all data points where Wind Speed > 10 km/h. 
Identify the Station Name (name_en) with the highest average Gust Factor. 
(High gust factors often indicate turbulence caused by skyscrapers/terrain).

Question 4: [Environmental Lapse Rate]
Calculate the correlation coefficient between 'Altitude' and average yearly 'Temperature' across all stations. 
This verifies the atmospheric lapse rate (temperature dropping with height).

Question 5: [Extreme Weather Impact]
Identify the single date (YYYY-MM-DD) that had the highest count of "Heat Stress Events" across all stations.
A "Heat Stress Event" is defined as a single observation where Temperature > 33°C AND Humidity > 75%.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd
import numpy as np

# Load the HK Weather data
try:
    hk_weather = pd.read_csv("/home/luobeier/cave-bench/benchmarks/data_analysis/Meteorological/HK_Weather/hk_weather_2025.csv")
    hk_stations = pd.read_csv("/home/luobeier/cave-bench/benchmarks/data_analysis/Meteorological/HK_Weather/hk_weather_stations.csv")
except Exception as e:
    print(f"Error loading datasets: {e}")
    hk_weather = None
    hk_stations = None

correct_answers = {
    "q1_ratio": 1.4068, 
    "q2_diff": 7.21, 
    "q3_station": "Wetland Park", 
    "q3_avg_gust_factor": 2.331,
    "q4_corr": -0.9531, 
    "q4_slope": -0.007,
    "q5_count": 40,
    "q5_date": "2025-06-08" 
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation for Dry Haze Ratio
    monsoon_haze_ratio = runtime.get_variable("monsoon_haze_ratio")
    
    if not isinstance(monsoon_haze_ratio, (float, np.floating)):
         return ValidatorResult(success=False, message=f"Q1: Type mismatch. Expected float, got {type(monsoon_haze_ratio).__name__}")
    
    # Allow small tolerance for floating point calc
    if round(monsoon_haze_ratio, 4) == correct_answers["q1_ratio"]:
        return ValidatorResult(success=True, message="Q1: Monsoon Haze Ratio is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q1: Incorrect. Expected {correct_answers['q1_ratio']}, got {monsoon_haze_ratio}")

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation for UHI Intensity
    uhi_intensity = runtime.get_variable("uhi_intensity")
    
    if not isinstance(uhi_intensity, (float, np.floating)):
        return ValidatorResult(success=False, message=f"Q2: Type mismatch. Expected float, got {type(uhi_intensity).__name__}")
        
    if round(uhi_intensity, 2) == correct_answers["q2_diff"]:
        return ValidatorResult(success=True, message="Q2: Nighttime UHI Intensity is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q2: Incorrect. Expected {correct_answers['q2_diff']}, got {uhi_intensity}")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation for Gust Factor Station
    max_gust_station = runtime.get_variable("max_gust_station")
    avg_gust_factor = runtime.get_variable("avg_gust_factor")

    if not isinstance(max_gust_station, str):
        return ValidatorResult(success=False, message=f"Q3: Type mismatch. Expected string, got {type(max_gust_station).__name__}")
    if not isinstance(avg_gust_factor, float):
        return ValidatorResult(success=False, message=f"Q3: Type mismatch. Expected float, got {type(avg_gust_factor).__name__}")

    station_correct = max_gust_station.strip().lower() == correct_answers["q3_station"].lower()
    avg_gust_factor_correct = round(avg_gust_factor, 3) == correct_answers["q3_avg_gust_factor"]

    if station_correct and avg_gust_factor_correct:
        return ValidatorResult(success=True, message="Q3: Highest Gust Factor Station is correct.")
    else:
        return ValidatorResult(success=False, message=f"Q3: Incorrect. Expected {correct_answers['q3_station']}, got {max_gust_station}. Expected {correct_answers['q3_avg_gust_factor']}, got {avg_gust_factor}")

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation for Lapse Rate Correlation
    altitude_temp_corr = runtime.get_variable("altitude_temp_corr")
    slope = runtime.get_variable("slope")
    
    if not isinstance(altitude_temp_corr, (float, np.floating)):
         return ValidatorResult(success=False, message=f"Q4: Type mismatch. Expected float, got {type(altitude_temp_corr).__name__}")
    if not isinstance(slope, (float, np.floating)):
         return ValidatorResult(success=False, message=f"Q4: Type mismatch. Expected float, got {type(slope).__name__}")

    corr_correct = round(altitude_temp_corr, 4) == correct_answers["q4_corr"]
    slope_correct = round(slope, 3) == correct_answers["q4_slope"]

    if corr_correct and slope_correct:
        return ValidatorResult(success=True, message="Q4: Altitude-Temperature correlation and slope are correct.")
    else:
        return ValidatorResult(success=False, message=f"Q4: Incorrect. Expected {correct_answers['q4_corr']}, got {altitude_temp_corr}. Expected {correct_answers['q4_slope']}, got {slope}")


def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    # Validation for Extreme Heat Date
    max_heat_date = runtime.get_variable("max_heat_date")
    heat_stress_count = runtime.get_variable("heat_stress_count")

    if not isinstance(heat_stress_count, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q5: Type mismatch. Expected int, got {type(heat_stress_count).__name__}")
    
    # Convert date to string if it's a date object
    if hasattr(max_heat_date, 'strftime'):
        # It's a date or datetime object
        max_heat_date_str = max_heat_date.strftime("%Y-%m-%d")
    elif isinstance(max_heat_date, str):
        max_heat_date_str = max_heat_date
    else:
        max_heat_date_str = str(max_heat_date)

    count_correct = heat_stress_count == correct_answers["q5_count"]
    date_correct = max_heat_date_str == correct_answers["q5_date"]

    if count_correct and date_correct:
        return ValidatorResult(success=True, message="Q5: Heat stress count and date are correct.")
    else:
        return ValidatorResult(success=False, message=f"Q5: Incorrect. Expected {correct_answers['q5_count']}, got {heat_stress_count}. Expected {correct_answers['q5_date']}, got {max_heat_date_str}")

variables = [
    # Input Datasets
    Variable(
        name="hk_weather",
        value=hk_weather,
        description="""DataFrame containing 10-minute weather data. 
        Columns and their dtypes: 
        - id:int64 (Unique ID)
        - station_id: int64 (Station ID, foreign key to hk_stations.id)
        - temperature: float64 (Temperature in Celsius)
        - visibility: float64 (Visibility in meters)
        - pressure: float64 (Pressure in hPa)
        - humidity: float64 (Humidity in percentage)
        - wind_degree: int64 (Wind degree in degrees) (0-360)
        - wind_speed: float64 (Wind speed in km/h)
        - wind_gust: float64 (Wind gust in km/h)
        - datetime: object (string, format: "YYYY-MM-DD HH:MM:SS") (10-minute interval)
        example: pd.DataFrame(
            {
                "id": [82483],
                "station_id": [11],
                "temperature": [18.8],
                "visibility": [NaN],
                "pressure": [NaN],
                "humidity": [73.0],
                "wind_degree": [NaN],
                "wind_speed": [NaN],
                "wind_gust": [NaN],
                "datetime": ["2025-01-01 00:00:00"]
            }
        )
        """
    ),
    Variable(
        name="hk_stations",
        value=hk_stations,
        description="""DataFrame containing station metadata. 
        Columns and their dtypes:
        - id: int64 (Unique ID)
        - name_en: object (string)
        - station_code: object (string)
        - latitude: float64 (Latitude)
        - longitude: float64 (Longitude)
        - altitude: float64 (Altitude in meters)
        - station_type: object (string)
        example: pd.DataFrame(
            {
                "id": [1],
                "name_en": ["Central Pier"],
                "station_code": ["CP1"],
                "latitude": [22.288889],
                "longitude": [114.155833],
                "altitude": [19.0],
                "station_type": ["AUTOMATIC_WEATHER_STATION"]
            }
        )
        """
    ),

    # Output Variables for Agents
    # Q1 Variables
    Variable(
        name="monsoon_haze_ratio",
        value=0.0,
        description="Float. Store the calculated ratio of Dry Haze probability (Winter Monsoon / Summer Monsoon) of question 1."
    ),
    # Q2 Variables
    Variable(
        name="uhi_intensity",
        value=0.0,
        description="Float. Store the calculated temperature difference between the hottest and coldest stations during nighttime (20:00-05:59) of question 2."
    ),
    # Q3 Variables
    Variable(
        name="max_gust_station",
        value="",
        description="String. Store the English name of the station with the highest average Gust Factor (Gust/Speed) of question 3."
    ),
    Variable(
        name="avg_gust_factor",
        value=0.0,
        description="Float. Store the average Gust Factor (Gust/Speed) for the station with the highest average Gust Factor of question 3."
    ),
    # Q4 Variables
    Variable(
        name="altitude_temp_corr",
        value=0.0,
        description="Float. Store the Pearson correlation coefficient between station altitude and average temperature of question 4."
    ),
    Variable(
        name="slope",
        value=0.0,
        description="Float. Store the slope of the linear regression line between station altitude and average temperature of question 4."
    ),
    # Q5 Variables
    Variable(
        name="heat_stress_count",
        value=0,
        description="Int. Store the number of Heat Stress Events (Temp>33, Hum>75) on the date with the highest frequency of question 5."
    ),
    Variable(
        name="max_heat_date",
        value="",
        description="String. Store the date (YYYY-MM-DD) with the highest frequency of Heat Stress Events (Temp>33, Hum>75) of question 5."
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