"""
This scenario is based on analyzing the red wine quality dataset.
The goal is to use pandas and data analysis techniques to progressively analyze
wine quality patterns and chemical characteristics.

Dataset Structure:
red_wine_quality.csv contains the following columns:
- fixed acidity: Fixed acids in wine (tartaric acid - g/dm³)
- volatile acidity: Volatile acids in wine (acetic acid - g/dm³)  
- citric acid: Citric acid content (g/dm³)
- residual sugar: Residual sugar after fermentation (g/dm³)
- chlorides: Salt content (sodium chloride - g/dm³)
- free sulfur dioxide: Free SO2 (mg/dm³)
- total sulfur dioxide: Total SO2 (mg/dm³)
- density: Wine density (g/cm³)
- pH: Acidity level (0-14 scale)
- sulphates: Potassium sulphate content (g/dm³)
- alcohol: Alcohol percentage (% by volume)
- quality: Wine quality score (0-10, target variable)

Question 1: Basic quality distribution analysis
Calculate the distribution of wine quality scores and identify the most common quality rating.
Store the quality distribution as a pandas Series and find the mode (most frequent quality).

Question 2: Chemical composition analysis
Calculate the average alcohol content for different quality categories:
- High quality wines (quality >= 7)
- Medium quality wines (quality 5-6) 
- Low quality wines (quality <= 4)

Question 3: Correlation analysis for key factors
Calculate correlation coefficients between quality and the top 3 chemical factors:
alcohol, volatile acidity, and sulphates. Determine which factor has the strongest
correlation with wine quality.

Question 4: Quality prediction using alcohol and acidity
Create a binary classification: wines with quality >= 6 are "good", others are "poor".
Calculate the mean alcohol content and volatile acidity for each class, then determine
accuracy if we predict "good" for wines with alcohol > mean_alcohol_good AND 
volatile_acidity < mean_volatile_acidity_good.

Question 5: Advanced multi-factor analysis
Perform a comprehensive analysis combining multiple chemical factors:
1. Create quality tiers: Low (3-4), Medium (5-6), High (7-8)
2. For each tier, calculate average values for alcohol, volatile acidity, citric acid
3. Find the chemical profile (tier) with the highest average alcohol content
4. Calculate what percentage of high-tier wines have above-average alcohol AND below-average volatile acidity
"""

from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn
import pandas as pd
from typing import List
from core.types import ToolCall

# Load the red wine quality dataset
wine_df = pd.read_csv("benchmarks/data_analysis/RedWineQuality/red_wine_quality.csv")

tools = []

# Correct answers (calculated from actual red wine dataset)
correct_answers = {
    "q1_most_common_quality": 5,  # Most frequent quality score
    "q1_quality_5_count": 681,   # Count of wines with quality 5
    "q2_high_quality_alcohol": 11.52,  # Average alcohol for quality >= 7
    "q2_medium_quality_alcohol": 10.26,  # Average alcohol for quality 5-6
    "q2_low_quality_alcohol": 10.22,   # Average alcohol for quality <= 4
    "q3_alcohol_correlation": 0.4762,   # Correlation between alcohol and quality
    "q3_volatile_acidity_correlation": -0.3906,  # Correlation between volatile acidity and quality
    "q3_strongest_factor": "alcohol",   # Factor with strongest correlation
    "q4_prediction_accuracy": 0.6079,  # Accuracy of simple prediction model
    "q5_highest_alcohol_tier": "High", # Tier with highest average alcohol
    "q5_high_tier_percentage": 70.05   # Percentage of high-tier wines meeting criteria
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the quality distribution analysis.
    """
    quality_distribution = runtime.get_variable_value("quality_distribution")
    most_common_quality = runtime.get_variable_value("most_common_quality")
    
    if quality_distribution is None or most_common_quality is None:
        return ValidatorResult(
            success=False,
            message="Variables 'quality_distribution' or 'most_common_quality' not found."
        )
    
    # Check if most common quality is correct
    if most_common_quality != correct_answers["q1_most_common_quality"]:
        return ValidatorResult(
            success=False,
            message=f"Incorrect most common quality. Expected {correct_answers['q1_most_common_quality']}, got {most_common_quality}."
        )
    
    # Check if quality 5 count is approximately correct (allow 5% tolerance)
    if isinstance(quality_distribution, pd.Series):
        quality_5_count = quality_distribution.get(5, 0)
        expected_count = correct_answers["q1_quality_5_count"]
        tolerance = expected_count * 0.05
        
        if abs(quality_5_count - expected_count) > tolerance:
            return ValidatorResult(
                success=False,
                message=f"Incorrect count for quality 5. Expected ~{expected_count}, got {quality_5_count}."
            )
    
    return ValidatorResult(
        success=True,
        message=f"Correct. Most common quality is {most_common_quality}."
    )

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the chemical composition analysis by quality categories.
    """
    avg_alcohol_high = runtime.get_variable_value("avg_alcohol_high")
    avg_alcohol_medium = runtime.get_variable_value("avg_alcohol_medium")
    avg_alcohol_low = runtime.get_variable_value("avg_alcohol_low")
    
    if any(var is None for var in [avg_alcohol_high, avg_alcohol_medium, avg_alcohol_low]):
        return ValidatorResult(
            success=False,
            message="One or more alcohol average variables not found."
        )
    
    tolerance = 0.05
    
    high_correct = abs(avg_alcohol_high - correct_answers["q2_high_quality_alcohol"]) <= tolerance
    medium_correct = abs(avg_alcohol_medium - correct_answers["q2_medium_quality_alcohol"]) <= tolerance
    low_correct = abs(avg_alcohol_low - correct_answers["q2_low_quality_alcohol"]) <= tolerance
    
    if high_correct and medium_correct and low_correct:
        return ValidatorResult(
            success=True,
            message=f"Correct. Alcohol averages - High: {avg_alcohol_high:.2f}%, Medium: {avg_alcohol_medium:.2f}%, Low: {avg_alcohol_low:.2f}%."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect alcohol averages. Check calculations for different quality categories."
        )

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the correlation analysis for key factors.
    """
    alcohol_correlation = runtime.get_variable_value("alcohol_correlation")
    volatile_acidity_correlation = runtime.get_variable_value("volatile_acidity_correlation")
    strongest_correlation_factor = runtime.get_variable_value("strongest_correlation_factor")
    
    if any(var is None for var in [alcohol_correlation, volatile_acidity_correlation, strongest_correlation_factor]):
        return ValidatorResult(
            success=False,
            message="One or more correlation variables not found."
        )
    
    tolerance = 0.02
    
    alcohol_correct = abs(alcohol_correlation - correct_answers["q3_alcohol_correlation"]) <= tolerance
    acidity_correct = abs(volatile_acidity_correlation - correct_answers["q3_volatile_acidity_correlation"]) <= tolerance
    strongest_correct = strongest_correlation_factor.lower() == correct_answers["q3_strongest_factor"].lower()
    
    if alcohol_correct and acidity_correct and strongest_correct:
        return ValidatorResult(
            success=True,
            message=f"Correct. Strongest correlation factor is {strongest_correlation_factor}."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect correlation analysis. Check correlation calculations."
        )

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the quality prediction model accuracy.
    """
    mean_alcohol_good = runtime.get_variable_value("mean_alcohol_good")
    mean_volatile_acidity_good = runtime.get_variable_value("mean_volatile_acidity_good")
    prediction_accuracy = runtime.get_variable_value("prediction_accuracy")
    
    if any(var is None for var in [mean_alcohol_good, mean_volatile_acidity_good, prediction_accuracy]):
        return ValidatorResult(
            success=False,
            message="One or more prediction variables not found."
        )
    
    tolerance = 0.02
    expected_accuracy = correct_answers["q4_prediction_accuracy"]
    
    accuracy_correct = abs(prediction_accuracy - expected_accuracy) <= tolerance
    
    if accuracy_correct:
        return ValidatorResult(
            success=True,
            message=f"Correct. Prediction accuracy is {prediction_accuracy:.4f}."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect prediction accuracy. Expected ~{expected_accuracy:.4f}, got {prediction_accuracy:.4f}."
        )

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the advanced multi-factor analysis.
    """
    tier_alcohol_averages = runtime.get_variable_value("tier_alcohol_averages")
    highest_alcohol_tier = runtime.get_variable_value("highest_alcohol_tier")
    high_tier_criteria_percentage = runtime.get_variable_value("high_tier_criteria_percentage")
    
    if any(var is None for var in [tier_alcohol_averages, highest_alcohol_tier, high_tier_criteria_percentage]):
        return ValidatorResult(
            success=False,
            message="One or more multi-factor analysis variables not found."
        )
    
    tier_correct = highest_alcohol_tier.lower() == correct_answers["q5_highest_alcohol_tier"].lower()
    
    tolerance = 0.1
    if high_tier_criteria_percentage < 1:
        high_tier_criteria_percentage = high_tier_criteria_percentage * 100
    percentage_correct = abs(high_tier_criteria_percentage - correct_answers["q5_high_tier_percentage"]) <= tolerance
    
    if tier_correct and percentage_correct:
        return ValidatorResult(
            success=True,
            message=f"Correct. Highest alcohol tier: {highest_alcohol_tier}, High-tier criteria percentage: {high_tier_criteria_percentage:.2f}%."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect multi-factor analysis. Check tier calculations and percentage computation."
        )

variables = [
    Variable(
        name="wine_df",
        value=wine_df,
        description="""A pandas DataFrame containing red wine quality data.
        Columns and their dtypes:
        - fixed acidity: float64
        - volatile acidity: float64
        - citric acid: float64
        - residual sugar: float64
        - chlorides: float64
        - free sulfur dioxide: float64
        - total sulfur dioxide: float64
        - density: float64
        - pH: float64
        - sulphates: float64
        - alcohol: float64
        - quality: int64 (target variable, range 3-8)

        Example: pd.DataFrame(data={
            "fixed acidity": [7.4],
            "volatile acidity": [0.7],
            "citric acid": [0.0],
            "residual sugar": [1.9],
            "chlorides": [0.076],
            "free sulfur dioxide": [11.0],
            "total sulfur dioxide": [34.0],
            "density": [0.9978],
            "pH": [3.51],
            "sulphates": [0.56],
            "alcohol": [9.4],
            "quality": [5]
        })
        
        You should use this variable to answer all questions.
        """
    ),
    Variable(
        name="quality_distribution",
        value=pd.Series(dtype=int),
        description="Stores the distribution of wine quality scores as a pandas Series."
    ),
    Variable(
        name="most_common_quality",
        value=0,
        description="Stores the most frequently occurring quality score."
    ),
    Variable(
        name="avg_alcohol_high",
        value=0.0,
        description="Stores average alcohol content for high quality wines (>= 7)."
    ),
    Variable(
        name="avg_alcohol_medium",
        value=0.0,
        description="Stores average alcohol content for medium quality wines (5-6)."
    ),
    Variable(
        name="avg_alcohol_low",
        value=0.0,
        description="Stores average alcohol content for low quality wines (<= 4)."
    ),
    Variable(
        name="alcohol_correlation",
        value=0.0,
        description="Stores correlation coefficient between alcohol and quality."
    ),
    Variable(
        name="volatile_acidity_correlation",
        value=0.0,
        description="Stores correlation coefficient between volatile acidity and quality."
    ),
    Variable(
        name="strongest_correlation_factor",
        value="",
        description="Stores the name of the factor with strongest correlation to quality."
    ),
    Variable(
        name="mean_alcohol_good",
        value=0.0,
        description="Stores mean alcohol content for good wines (quality >= 6)."
    ),
    Variable(
        name="mean_volatile_acidity_good",
        value=0.0,
        description="Stores mean volatile acidity for good wines (quality >= 6)."
    ),
    Variable(
        name="prediction_accuracy",
        value=0.0,
        description="Stores accuracy of the simple prediction model."
    ),
    Variable(
        name="tier_alcohol_averages",
        value={},
        description="Stores average alcohol content for each quality tier (Low, Medium, High)."
    ),
    Variable(
        name="highest_alcohol_tier",
        value="",
        description="Stores the quality tier with highest average alcohol content."
    ),
    Variable(
        name="high_tier_criteria_percentage",
        value=0.0,
        description="Stores percentage of high-tier wines meeting the dual criteria."
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