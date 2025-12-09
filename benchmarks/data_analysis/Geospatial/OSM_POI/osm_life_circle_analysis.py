"""
This scenario is based on analyzing OpenStreetMap (OSM) Point of Interest (POI) data.
The goal is to perform deep spatial and sociological analysis using pandas and scipy,
moving beyond simple aggregations to calculate indices like Diversity Entropy, 
Food Swamp Index, and Spatial Competition.

Dataset Structure:
osm_pois.csv contains:
- id: Unique identifier
- name: Name of the POI
- category: Type of POI ('supermarket', 'fast_food', 'cafe', 'school', 'park', 'hospital')
- lat: Latitude
- lon: Longitude
- district: The administrative district name

Question 1: Identify the district with the worst 'Food Swamp Index'. 
            Calculate the index for each district as: Count(Fast Food) / (Count(Fast Food) + Count(Supermarket)).
            Return the district name with the highest index and the index value itself as a tuple.
Question 2: Calculate the Shannon Entropy (Diversity Index) of POI categories for the 'Central and Western' district.
            Higher entropy indicates a more mixed/diverse functional area.
Question 3: Analyze spatial competition for Coffee Shops. Calculate the Average Nearest Neighbor (ANN) distance 
            (in meters) for all POIs with category 'cafe'.
Question 4: Identify the specific POI (return its ID) that is the "Most Competitive Node". 
            This is defined as the POI with the highest number of *same-category* competitors within a 0.005 degree radius.
Question 5: Identify the school with the highest 'Fast Food Exposure Score'.
            For each school, find all fast_food locations within 0.005 degrees radius.
            Convert coordinates to local meter coordinates: lon_scale = cos(latitude_mean) * 111320, lat_scale = 111320.
            Calculate distances in meters and the score as the sum of (1 / distance_in_meters) for these nearby fast_food locations.
            Return the School ID and the Score as a tuple.
"""

from typing import List, Tuple
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd
import numpy as np

# Load the dataset
try:
    osm_df = pd.read_csv("benchmarks/data_analysis/Geospatial/OSM_POI/osm_pois_hk.csv")
except Exception as e:
    print(f"Error loading OSM dataset: {e}")
    osm_df = None


correct_answers = {
    "q1_district": "Sham Shui Po", 
    "q1_index": 0.8099,
    "q2_entropy": 1.5965,          
    "q3_ann_distance": 232.39,    
    "q4_competitive_id": 1598,    
    "q5_school_id": 1634,          
    "q5_score": 0.2841       
}

# --- Validators ---

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    worst_food_swamp = runtime.get_variable("worst_food_swamp")

    # Check type: Should be a tuple (str, float)
    if not isinstance(worst_food_swamp, tuple) or len(worst_food_swamp) != 2:
        return ValidatorResult(success=False, message=f"Q1: Expected a tuple (district_name, index_value), got {type(worst_food_swamp)}")
    
    district, index_val = worst_food_swamp
    
    if not isinstance(district, str) or not isinstance(index_val, float):
        return ValidatorResult(success=False, message=f"Q1: Tuple contents should be (str, float). Got ({type(district)}, {type(index_val)})")

    # Check values
    district_match = district == correct_answers["q1_district"]
    index_match = round(index_val, 4) == correct_answers["q1_index"]

    if district_match and index_match:
        return ValidatorResult(success=True, message="Q1: Correct.")
    return ValidatorResult(success=False, message=f"Q1: Incorrect. Expected ({correct_answers['q1_district']}, {correct_answers['q1_index']}), got {worst_food_swamp}")

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    diversity_entropy = runtime.get_variable("diversity_entropy")

    if not isinstance(diversity_entropy, float):
        return ValidatorResult(success=False, message=f"Q2: Expected float, got {type(diversity_entropy).__name__}")
    
    if round(diversity_entropy, 4) == correct_answers["q2_entropy"]:
        return ValidatorResult(success=True, message="Q2: Correct.")
    return ValidatorResult(success=False, message=f"Q2: Expected {correct_answers['q2_entropy']}, got {diversity_entropy}")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    avg_nn_dist = runtime.get_variable("avg_nn_dist")

    if not isinstance(avg_nn_dist, float):
        return ValidatorResult(success=False, message=f"Q3: Expected float, got {type(avg_nn_dist).__name__}")

    if round(avg_nn_dist, 2) == correct_answers["q3_ann_distance"]:
        return ValidatorResult(success=True, message="Q3: Correct.")
    return ValidatorResult(success=False, message=f"Q3: Expected {correct_answers['q3_ann_distance']}, got {avg_nn_dist}")

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    most_competitive_id = runtime.get_variable("most_competitive_id")

    # ID can be int or int64
    if not isinstance(most_competitive_id, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q4: Expected int, got {type(most_competitive_id).__name__}")

    if most_competitive_id == correct_answers["q4_competitive_id"]:
        return ValidatorResult(success=True, message="Q4: Correct.")
    return ValidatorResult(success=False, message=f"Q4: Expected {correct_answers['q4_competitive_id']}, got {most_competitive_id}")

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    riskiest_school = runtime.get_variable("riskiest_school")
    
    # Check type: Tuple (int, float)
    if not isinstance(riskiest_school, tuple) or len(riskiest_school) != 2:
        return ValidatorResult(success=False, message=f"Q5: Expected a tuple (school_id, score), got {type(riskiest_school)}")
    
    s_id, score = riskiest_school
    
    if not isinstance(s_id, (int, np.integer)) or not isinstance(score, float):
         return ValidatorResult(success=False, message=f"Q5: Tuple contents should be (int, float).")

    id_match = s_id == correct_answers["q5_school_id"]
    score_match = round(score, 4) == correct_answers["q5_score"]

    if id_match and score_match:
        return ValidatorResult(success=True, message="Q5: Correct.")
    return ValidatorResult(success=False, message=f"Q5: Incorrect. Expected ID {correct_answers['q5_school_id']} with score {correct_answers['q5_score']}, got ID {s_id} with score {score}")


variables = [
    Variable(
        name="osm_df",
        value=osm_df,
        description="""
        A pandas DataFrame containing OSM POI data.
        Columns and their dtypes:
        - id: int64 (Unique ID)
        - name: object (string, Name of the place)
        - category: object (string, e.g., 'fast_food', 'supermarket', 'cafe', 'school', 'hospital', 'park', etc.)
        - lat: float64 (Latitude)
        - lon: float64 (Longitude)
        - district: object (string, District name, e.g., 'Central and Western', 'Sham Shui Po')
        Example: pd.DataFrame(data={
            "id": [1],
            "name": ["ParkNShop (Peak Galleria)"],
            "category": ["supermarket"],
            "lat": [22.270048],
            "lon": [114.150153],
            "district": ["Central and Western"],
        })
        """
    ),
    Variable(
        name="worst_food_swamp",
        value=("", 0.0),
        description="Tuple (str, float). The name of the district with the highest Food Swamp Index and the index value."
    ),
    Variable(
        name="diversity_entropy",
        value=0.0,
        description="Float. Shannon Entropy of category distribution in Central and Western, unit: nat."
    ),
    Variable(
        name="avg_nn_dist",
        value=0.0,
        description="Float. Average distance to the nearest neighbor for 'cafe' category, unit: meters."
    ),
    Variable(
        name="most_competitive_id",
        value=0,
        description="Int. The ID of the POI with the most same-category neighbors within 0.005 radius."
    ),
    Variable(
        name="riskiest_school",
        value=(0, 0.0),
        description="Tuple (int, float). The ID of the school with highest Fast Food Exposure Score and the score itself. The score is calculated as the sum of (1 / distance_in_meters) using local meter coordinates (lon_scale = cos(lat_mean) * 111320, lat_scale = 111320)."
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