"""
This scenario is based on analyzing the famous Titanic dataset.
The goal is to use pandas and data analysis techniques to progressively analyze
passenger survival patterns and demographic characteristics.

Dataset Structure:
titanic.csv contains the following columns:
- PassengerId: Unique identifier for each passenger
- Survived: Survival status (0 = No, 1 = Yes)
- Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Name: Passenger name
- Sex: Gender (male/female)
- Age: Age in years
- SibSp: Number of siblings/spouses aboard
- Parch: Number of parents/children aboard
- Ticket: Ticket number
- Fare: Passenger fare
- Cabin: Cabin number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Question 1: Calculate survival rate by passenger class
Calculate the survival rate (percentage of survivors) for each passenger class (Pclass).
Store the result as a pandas Series with Pclass as index and survival rate as values.

Question 2: Find average age of survivors vs non-survivors
Calculate the average age for passengers who survived and those who didn't survive.
Handle missing age values appropriately by excluding them from the calculation.

Question 3: Analyze survival patterns by gender and class
Create a cross-tabulation showing survival counts and survival rates for each combination 
of gender (Sex) and passenger class (Pclass). Calculate both absolute numbers and percentages.

Question 4: Identify fare distribution patterns for survivors
For passengers who survived, analyze the fare distribution by passenger class and gender.
Calculate the median fare, 25th percentile, and 75th percentile for each group.
Focus on passengers with non-null fare values.

Question 5: Complex family survival analysis
Analyze survival patterns based on family size (SibSp + Parch + 1, including the passenger).
Create family size categories: Solo (1), Small (2-4), Large (5+), then calculate:
- Overall survival rate by family size category
- Survival rate by family size category and passenger class
- Determine which combination of family size and class had the highest survival rate
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd
import re

# Load the Titanic training dataset (contains both features and survival labels)
titanic_df = pd.read_csv("benchmarks/data_analysis/Titanic/train.csv")

tools = []

# Correct answers (calculated from actual Titanic dataset)
correct_answers = {
    "q1_class1": 0.6296,  # Survival rate for 1st class (approximately 62.96%)
    "q1_class2": 0.4728,  # Survival rate for 2nd class (approximately 47.28%)
    "q1_class3": 0.2424,  # Survival rate for 3rd class (approximately 24.24%)
    "q2_survived_avg_age": 28.34,   # Average age of survivors
    "q2_not_survived_avg_age": 30.63,  # Average age of non-survivors
    "q3_male_class1_survival_rate": 0.3689,  # Male 1st class survival rate
    "q3_female_class3_survival_rate": 0.5000,  # Female 3rd class survival rate
    "q4_median_fare_female_class1": 82.1708,  # Median fare for female 1st class survivors
    "q4_q75_fare_male_class2": 26.0000,  # 75th percentile fare for male 2nd class survivors
    "q5_small_family_survival_rate": 0.5788,  # Survival rate for small families (2-4)
    "q5_best_combination": "Large_2",  # Best survival combination: Large family + 2nd class
    "q5_best_survival_rate": 1.0000  # Survival rate for the best combination
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the survival rate by passenger class.
    """
    survival_rate_by_class = runtime.get_variable_value("survival_rate_by_class")
    
    if survival_rate_by_class is None:
        return ValidatorResult(
            success=False,
            message="Variable 'survival_rate_by_class' not found."
        )
    
    # Check if it's a pandas Series with correct structure
    if not isinstance(survival_rate_by_class, pd.Series):
        return ValidatorResult(
            success=False,
            message="Expected a pandas Series for survival_rate_by_class."
        )
    
    # Handle percentage values (convert to decimal if > 1)
    survival_rate_by_class = survival_rate_by_class.copy()
    if (survival_rate_by_class.values > 1).any():
        survival_rate_by_class = survival_rate_by_class / 100.0
    
    # Check values with tolerance
    tolerance = 0.01
    checks = [
        (1, correct_answers["q1_class1"]),
        (2, correct_answers["q1_class2"]),
        (3, correct_answers["q1_class3"])
    ]
    
    for pclass, expected in checks:
        if pclass not in survival_rate_by_class.index:
            return ValidatorResult(
                success=False,
                message=f"Missing class {pclass} in survival_rate_by_class."
            )
        
        actual = survival_rate_by_class[pclass]
        if abs(actual - expected) > tolerance:
            return ValidatorResult(
                success=False,
                message=f"Incorrect survival rate for class {pclass}. Expected ~{expected:.4f}, got {actual:.4f}."
            )
    
    return ValidatorResult(
        success=True,
        message=f"Correct. Survival rates by class calculated successfully."
    )

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the average age of survivors vs non-survivors.
    """
    avg_age_survived = runtime.get_variable_value("avg_age_survived")
    avg_age_not_survived = runtime.get_variable_value("avg_age_not_survived")
    
    if avg_age_survived is None or avg_age_not_survived is None:
        return ValidatorResult(
            success=False,
            message="Variables 'avg_age_survived' or 'avg_age_not_survived' not found."
        )
    
    tolerance = 0.5
    expected_survived = correct_answers["q2_survived_avg_age"]
    expected_not_survived = correct_answers["q2_not_survived_avg_age"]
    
    survived_correct = abs(avg_age_survived - expected_survived) <= tolerance
    not_survived_correct = abs(avg_age_not_survived - expected_not_survived) <= tolerance
    
    if survived_correct and not_survived_correct:
        return ValidatorResult(
            success=True,
            message=f"Correct. Average age - Survivors: {avg_age_survived:.2f}, Non-survivors: {avg_age_not_survived:.2f}."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect averages. Expected survivors: ~{expected_survived:.2f}, got {avg_age_survived:.2f}. Expected non-survivors: ~{expected_not_survived:.2f}, got {avg_age_not_survived:.2f}."
        )

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the survival analysis by gender and class.
    """
    survival_crosstab = runtime.get_variable_value("survival_crosstab")
    survival_rates_by_gender_class = runtime.get_variable_value("survival_rates_by_gender_class")
    
    if survival_crosstab is None or survival_rates_by_gender_class is None:
        return ValidatorResult(
            success=False,
            message="Variables 'survival_crosstab' or 'survival_rates_by_gender_class' not found."
        )
    
    # Check specific survival rates
    tolerance = 0.02
    
    try:
        # Check male 1st class survival rate
        rates_df = survival_rates_by_gender_class.copy()
        if (rates_df.values > 1).any():
            rates_df = rates_df / 100.0

        # Extract scalar values directly
        male_class1_rate = float(rates_df.loc['male', 1])
        female_class3_rate = float(rates_df.loc['female', 3])
        
        male_correct = abs(male_class1_rate - correct_answers["q3_male_class1_survival_rate"]) <= tolerance
        female_correct = abs(female_class3_rate - correct_answers["q3_female_class3_survival_rate"]) <= tolerance
        
        if male_correct and female_correct:
            return ValidatorResult(
                success=True,
                message="Correct. Cross-tabulation analysis completed successfully."
            )
        else:
            return ValidatorResult(
                success=False,
                message=f"Incorrect survival rates in cross-tabulation analysis."
            )
    except (KeyError, IndexError, AttributeError, TypeError, ValueError) as e:
        return ValidatorResult(
            success=False,
            message=f"Error accessing survival rates. Check the structure of survival_rates_by_gender_class. Error: {str(e)}"
        )

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the fare distribution analysis for survivors.
    """
    fare_stats_survivors = runtime.get_variable_value("fare_stats_survivors")
    
    if fare_stats_survivors is None:
        return ValidatorResult(
            success=False,
            message="Variable 'fare_stats_survivors' not found."
        )
    
    tolerance = 2.0
    
    try:
        if isinstance(fare_stats_survivors, pd.DataFrame):
            median_col = '50%'
            q3_col = '75%'

            if median_col not in fare_stats_survivors.columns or q3_col not in fare_stats_survivors.columns:
                 return ValidatorResult(success=False, message="Could not find median/Q3 columns.")

            female_class1_median = fare_stats_survivors.loc[('female', 1), median_col]
            male_class2_q75 = fare_stats_survivors.loc[('male', 2), q3_col]
        
        median_correct = abs(female_class1_median - correct_answers["q4_median_fare_female_class1"]) <= tolerance
        q75_correct = abs(male_class2_q75 - correct_answers["q4_q75_fare_male_class2"]) <= tolerance
        
        if median_correct and q75_correct:
            return ValidatorResult(
                success=True,
                message="Correct. Fare distribution analysis completed successfully."
            )
        else:
            return ValidatorResult(
                success=False,
                message="Incorrect fare statistics in the analysis."
            )
    except (KeyError, IndexError, AttributeError):
        return ValidatorResult(
            success=False,
            message="Error accessing fare statistics. Check the structure of fare_stats_survivors."
        )

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the complex family survival analysis.
    """
    family_survival_rates = runtime.get_variable_value("family_survival_rates")
    family_class_survival_rates = runtime.get_variable_value("family_class_survival_rates")
    best_combination = runtime.get_variable_value("best_combination")
    best_survival_rate = runtime.get_variable_value("best_survival_rate")

    if any(var is None for var in [family_survival_rates, family_class_survival_rates, best_combination, best_survival_rate]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for family analysis."
        )

    # Clean up best_combination string (after None check)
    best_combination = re.sub(r'(_\d+)[a-zA-Z].*$', r'\1', str(best_combination))

    tolerance = 0.02
    
    try:
        # Check small family survival rate
        if 'Small' in family_survival_rates:
            small_family_rate = family_survival_rates['Small']
            if small_family_rate  > 1:
                small_family_rate = small_family_rate / 100.0
            small_correct = abs(small_family_rate - correct_answers["q5_small_family_survival_rate"]) <= tolerance
        else:
            small_correct = False
        
        # Check best combination
        combination_correct = str(best_combination).strip() == correct_answers["q5_best_combination"]
        if best_survival_rate > 1:
            best_survival_rate = best_survival_rate / 100.0
        rate_correct = abs(best_survival_rate - correct_answers["q5_best_survival_rate"]) <= tolerance
        
        if small_correct and combination_correct and rate_correct:
            return ValidatorResult(
                success=True,
                message=f"Correct. Best survival combination: {best_combination} with rate {best_survival_rate:.4f}."
            )
        else:
            error_parts = []
            if not small_correct:
                error_parts.append("Small family survival rate incorrect")
            if not combination_correct:
                error_parts.append(f"Best combination expected {correct_answers['q5_best_combination']}, got {best_combination}")
            if not rate_correct:
                error_parts.append(f"Best survival rate expected ~{correct_answers['q5_best_survival_rate']:.4f}, got {best_survival_rate:.4f}")
            
            return ValidatorResult(
                success=False,
                message=f"Incorrect family analysis: {'; '.join(error_parts)}."
            )
    except (KeyError, IndexError, AttributeError) as e:
        return ValidatorResult(
            success=False,
            message=f"Error in family analysis validation: {str(e)}"
        )

variables = [
    Variable(
        name="titanic_df",
        value=titanic_df,
        description="""A pandas DataFrame from 'train.csv' containing Titanic passenger training data.
        Columns and their dtypes:
        - PassengerId: int64
        - Survived: int64 (0 = No, 1 = Yes)
        - Pclass: int64 (1 = 1st, 2 = 2nd, 3 = 3rd)
        - Name: object (string)
        - Sex: object (string, 'male'/'female')
        - Age: float64 (may contain NaN)
        - SibSp: int64 (siblings/spouses aboard)
        - Parch: int64 (parents/children aboard)
        - Ticket: object (string)
        - Fare: float64 (may contain NaN)
        - Cabin: object (string, many NaN)
        - Embarked: object (string, 'C'/'Q'/'S', may contain NaN)
        
        Example: pd.DataFrame(data={
            'PassengerId': [1],
            'Survived': [0],
            'Pclass': [3],
            'Name': ['Braund, Mr. Owen Harris'],
            'Sex': ['male'],
            'Age': [22.0],
            'SibSp': [1],
            'Parch': [0],
            'Ticket': ['A/5 21171'],
            'Fare': [7.25],
            'Cabin': [np.nan],
            'Embarked': ['S']
        })
        You should use this variable to answer question 1, 2, 3, 4 and 5.
        """
    ),
    Variable(
        name="survival_rate_by_class",
        value=pd.Series(dtype=float),
        description="Stores survival rate for each passenger class as a pandas Series."
    ),
    Variable(
        name="avg_age_survived",
        value=0.0,
        description="Stores the average age of passengers who survived."
    ),
    Variable(
        name="avg_age_not_survived", 
        value=0.0,
        description="Stores the average age of passengers who did not survive."
    ),
    Variable(
        name="survival_crosstab",
        value=pd.DataFrame(),
        description="Stores cross-tabulation of survival by gender and class."
    ),
    Variable(
        name="survival_rates_by_gender_class",
        value=pd.DataFrame(),
        description="Stores survival rates for each gender-class combination."
    ),
    Variable(
        name="fare_stats_survivors",
        value=pd.DataFrame(),
        description="Stores fare distribution statistics (median, Q1, Q3) for survivors by gender and class."
    ),
    Variable(
        name="family_survival_rates",
        value={},
        description="Stores survival rates by family size category (Solo, Small, Large)."
    ),
    Variable(
        name="family_class_survival_rates", 
        value=pd.DataFrame(),
        description="Stores survival rates by family size category and passenger class."
    ),
    Variable(
        name="best_combination",
        value="",
        description="Stores the family size and class combination with highest survival rate."
    ),
    Variable(
        name="best_survival_rate",
        value=0.0,
        description="Stores the highest survival rate found in family-class combinations."
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