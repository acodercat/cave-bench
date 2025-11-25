"""
This scenario is based on analyzing a college student placement dataset.
The goal is to use pandas and data analysis techniques to progressively analyze
student placement patterns and identify key factors affecting placement success.

Dataset Structure:
college_student_placement_dataset.csv contains the following columns:
- College_ID: Unique identifier for each college (string)
- IQ: Intelligence Quotient score (integer)
- Prev_Sem_Result: Previous semester result score (float)
- CGPA: Cumulative Grade Point Average (float)
- Academic_Performance: Academic performance rating 1-10 (integer)
- Internship_Experience: Whether student has internship experience (Yes/No)
- Extra_Curricular_Score: Extra-curricular activities score 1-10 (integer)
- Communication_Skills: Communication skills rating 1-10 (integer)
- Projects_Completed: Number of projects completed (integer)
- Placement: Whether student got placed (Yes/No)

Question 1: Calculate overall placement rate
Calculate the overall placement rate (percentage of students who got placed).
Store the result as a float value representing the percentage.

Question 2: Find average CGPA and IQ for placed vs non-placed students
Calculate the average CGPA and IQ for students who got placed and those who didn't.
Handle missing values appropriately by excluding them from calculations.

Question 3: Analyze placement patterns by internship experience and academic performance
Create academic performance categories: Low (1-3), Medium (4-7), High (8-10).
Create a cross-tabulation showing placement counts and placement rates for each combination 
of internship experience and academic performance level.

Question 4: Analyze skills distribution for placed students
For students who got placed, analyze the distribution of communication skills and 
extra-curricular scores by internship experience. Calculate the median, 25th percentile, 
and 75th percentile for each group.

Question 5: Comprehensive competency analysis
Create an overall competency score using weighted average:
- CGPA * 0.3 + IQ/100 * 0.2 + Academic_Performance * 0.2 + Communication_Skills * 0.15 + Extra_Curricular_Score * 0.15
Categorize students: High (>7), Medium (5-7), Low (<5)
Calculate placement rates by competency level and internship experience, 
then find the combination with highest placement rate.
"""

from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn
import pandas as pd
import re
from typing import List
from core.types import ToolCall

# Load the placement dataset
placement_df = pd.read_csv("benchmarks/data_analysis/Placement/college_student_placement_dataset.csv")

tools = []

# Correct answers (calculated from actual placement dataset)
# These values should be updated based on the actual dataset analysis
correct_answers = {
    "q1_overall_rate": 16.59,  # Overall placement rate (approximate)
    "q2_cgpa_placed": 8.59,   # Average CGPA of placed students
    "q2_cgpa_not_placed": 7.32,  # Average CGPA of non-placed students
    "q2_iq_placed": 109.12,   # Average IQ of placed students
    "q2_iq_not_placed": 97.55,  # Average IQ of non-placed students
    "q3_yes_high_rate": 0.1708,  # Placement rate for Yes internship + High performance
    "q3_no_medium_rate": 0.1705,  # Placement rate for No internship + Medium performance
    "q4_median_comm_yes": 9.0,  # Median communication skills for placed with internship
    "q4_q75_extra_no": 8.0,  # 75th percentile extra-curricular for placed without internship
    "q5_high_internship_rate": 0.6034,  # Placement rate for High competency students
    "q5_best_combination": "High_No",  # Best combination: High competency + Yes internship
    "q5_best_rate": 0.6176  # Best placement rate
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the overall placement rate calculation.
    """
    overall_placement_rate = runtime.get_variable_value("overall_placement_rate")
    
    if overall_placement_rate is None:
        return ValidatorResult(
            success=False,
            message="Variable 'overall_placement_rate' not found."
        )
    
    # Convert to percentage if necessary
    if overall_placement_rate <= 1.0:
        overall_placement_rate = overall_placement_rate * 100
    
    # Check value with tolerance (assuming actual rate is around 45%)
    tolerance = 0.1
    expected = correct_answers["q1_overall_rate"]
    
    if abs(overall_placement_rate - expected) <= tolerance:
        return ValidatorResult(
            success=True,
            message=f"Correct. Overall placement rate: {overall_placement_rate:.2f}%."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect placement rate. Expected around {expected}%, got {overall_placement_rate:.2f}%."
        )

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the average CGPA and IQ for placed vs non-placed students.
    """
    variables = ["avg_cgpa_placed", "avg_cgpa_not_placed", "avg_iq_placed", "avg_iq_not_placed"]
    values = {}
    
    for var in variables:
        values[var] = runtime.get_variable_value(var)
        if values[var] is None:
            return ValidatorResult(
                success=False,
                message=f"Variable '{var}' not found."
            )
    
    tolerance = 0.1
    checks = [
        ("avg_cgpa_placed", "q2_cgpa_placed"),
        ("avg_cgpa_not_placed", "q2_cgpa_not_placed"),
        ("avg_iq_placed", "q2_iq_placed"),
        ("avg_iq_not_placed", "q2_iq_not_placed")
    ]
    
    all_correct = True
    error_messages = []
    
    for var_name, answer_key in checks:
        actual = values[var_name]
        expected = correct_answers[answer_key]
        if abs(actual - expected) > tolerance:
            all_correct = False
            error_messages.append(f"{var_name}: expected ~{expected:.2f}, got {actual:.2f}")
    
    if all_correct:
        return ValidatorResult(
            success=True,
            message="Correct. Average scores calculated successfully for both groups."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect averages: {'; '.join(error_messages)}."
        )

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the placement analysis by internship experience and academic performance.
    """
    placement_crosstab = runtime.get_variable_value("placement_crosstab")
    placement_rates_by_internship_performance = runtime.get_variable_value("placement_rates_by_internship_performance")
    
    if placement_crosstab is None or placement_rates_by_internship_performance is None:
        return ValidatorResult(
            success=False,
            message="Variables 'placement_crosstab' or 'placement_rates_by_internship_performance' not found."
        )
    
    tolerance = 0.05
    
    try:
        # Check specific placement rates
        rates_df = placement_rates_by_internship_performance.copy()
        if (rates_df.values > 1).any():
            rates_df = rates_df / 100.0
        
        # Check Yes internship + High performance rate
        if hasattr(rates_df, 'loc'):
            yes_high_rate = rates_df.loc['Yes', 'High']
        else:
            yes_high_rate = rates_df[('Yes', 'High')]
        
        # Check No internship + Medium performance rate
        if hasattr(rates_df, 'loc'):
            no_medium_rate = rates_df.loc['No', 'Medium']
        else:
            no_medium_rate = rates_df[('No', 'Medium')]
        
        yes_high_correct = abs(yes_high_rate - correct_answers["q3_yes_high_rate"]) <= tolerance
        no_medium_correct = abs(no_medium_rate - correct_answers["q3_no_medium_rate"]) <= tolerance
        
        if yes_high_correct and no_medium_correct:
            return ValidatorResult(
                success=True,
                message="Correct. Cross-tabulation analysis completed successfully."
            )
        else:
            return ValidatorResult(
                success=False,
                message="Incorrect placement rates in cross-tabulation analysis."
            )
    except (KeyError, IndexError, AttributeError):
        return ValidatorResult(
            success=False,
            message="Error accessing placement rates. Check the structure of placement_rates_by_internship_performance."
        )

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the skills distribution analysis for placed students.
    """
    skills_stats_placed = runtime.get_variable_value("skills_stats_placed")
    
    if skills_stats_placed is None:
        return ValidatorResult(
            success=False,
            message="Variable 'skills_stats_placed' not found."
        )
    
    try:
        if isinstance(skills_stats_placed, pd.DataFrame):
            median_col = '50%'
            q3_col = '75%'
            
            if median_col not in skills_stats_placed.columns or q3_col not in skills_stats_placed.columns:
                return ValidatorResult(success=False, message="Could not find median/Q3 columns.")
            
            # Check median communication skills for Yes internship
            comm_yes_median = skills_stats_placed.loc[('Yes', 'Communication_Skills'), median_col]
            
            # Check 75th percentile extra-curricular for No internship
            extra_no_q75 = skills_stats_placed.loc[('No', 'Extra_Curricular_Score'), q3_col]
        
        median_correct = comm_yes_median == correct_answers["q4_median_comm_yes"]
        q75_correct = extra_no_q75 == correct_answers["q4_q75_extra_no"]
        
        if median_correct and q75_correct:
            return ValidatorResult(
                success=True,
                message="Correct. Skills distribution analysis completed successfully."
            )
        else:
            return ValidatorResult(
                success=False,
                message="Incorrect skills statistics in the analysis."
            )
    except (KeyError, IndexError, AttributeError):
        return ValidatorResult(
            success=False,
            message="Error accessing skills statistics. Check the structure of skills_stats_placed."
        )

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the comprehensive competency analysis.
    """
    competency_placement_rates = runtime.get_variable_value("competency_placement_rates")
    competency_internship_placement_rates = runtime.get_variable_value("competency_internship_placement_rates")
    best_combination = runtime.get_variable_value("best_combination")
    best_placement_rate = runtime.get_variable_value("best_placement_rate")
    
    if any(var is None for var in [competency_placement_rates, competency_internship_placement_rates, 
                                   best_combination, best_placement_rate]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for competency analysis."
        )
    
    tolerance = 0.03
    
    try:
        # Clean best_combination string
        if isinstance(best_combination, tuple):
            best_combination = '_'.join(str(item) for item in best_combination)
        else:
            best_combination = re.sub(r'(_[YyNn][eo][s]?).*$', r'\1', str(best_combination).strip())
            best_combination = best_combination.replace('_yes', '_Yes').replace('_no', '_No')
        
        # Check high competency rate
        if ('High' in competency_internship_placement_rates.index) and ('Yes' in competency_internship_placement_rates.columns):
            high_competency_internship_rate = competency_internship_placement_rates.loc['High', 'Yes']
            if high_competency_internship_rate > 1:
                high_competency_internship_rate = high_competency_internship_rate / 100.0
            high_correct = abs(high_competency_internship_rate - correct_answers["q5_high_internship_rate"]) <= tolerance
        else:
            high_correct = False
        
        # Check best combination
        combination_correct = best_combination == correct_answers["q5_best_combination"]
        if best_placement_rate > 1:
            best_placement_rate = best_placement_rate / 100.0
        rate_correct = abs(best_placement_rate - correct_answers["q5_best_rate"]) <= tolerance
        
        if high_correct and combination_correct and rate_correct:
            return ValidatorResult(
                success=True,
                message=f"Correct. Best placement combination: {best_combination} with rate {best_placement_rate:.4f}."
            )
        else:
            error_parts = []
            if not high_correct:
                error_parts.append("High competency placement rate incorrect")
            if not combination_correct:
                error_parts.append(f"Best combination expected {correct_answers['q5_best_combination']}, got {best_combination}")
            if not rate_correct:
                error_parts.append(f"Best placement rate expected ~{correct_answers['q5_best_rate']:.4f}, got {best_placement_rate:.4f}")
            
            return ValidatorResult(
                success=False,
                message=f"Incorrect competency analysis: {'; '.join(error_parts)}."
            )
    except (KeyError, IndexError, AttributeError) as e:
        return ValidatorResult(
            success=False,
            message=f"Error in competency analysis validation: {str(e)}"
        )

variables = [
    Variable(
        name="placement_df",
        value=placement_df,
        description="""A pandas DataFrame from 'college_student_placement_dataset.csv' containing college student placement data.
        Columns and their dtypes:
        - College_ID: object (string)
        - IQ: int64 (Intelligence Quotient score)
        - Prev_Sem_Result: float64 (Previous semester result score)
        - CGPA: float64 (Cumulative Grade Point Average)
        - Academic_Performance: int64 (Academic performance rating 1-10)
        - Internship_Experience: object (string, 'Yes'/'No')
        - Extra_Curricular_Score: int64 (Extra-curricular activities score 1-10)
        - Communication_Skills: int64 (Communication skills rating 1-10)
        - Projects_Completed: int64 (Number of projects completed)
        - Placement: object (string, 'Yes'/'No')
        
        Example: pd.DataFrame(data={
            'College_ID': ['CLG0030'],
            'IQ': [107],
            'Prev_Sem_Result': [6.61],
            'CGPA': [6.28],
            'Academic_Performance': [8],
            'Internship_Experience': ['No'],
            'Extra_Curricular_Score': [8],
            'Communication_Skills': [8],
            'Projects_Completed': [4],
            'Placement': ['No']
        })
        You should use this variable to answer all questions.
        """
    ),
    Variable(
        name="overall_placement_rate",
        value=0.0,
        description="Stores the overall placement rate as a percentage."
    ),
    Variable(
        name="avg_cgpa_placed",
        value=0.0,
        description="Stores the average CGPA of students who got placed."
    ),
    Variable(
        name="avg_cgpa_not_placed", 
        value=0.0,
        description="Stores the average CGPA of students who did not get placed."
    ),
    Variable(
        name="avg_iq_placed",
        value=0.0,
        description="Stores the average IQ of students who got placed."
    ),
    Variable(
        name="avg_iq_not_placed", 
        value=0.0,
        description="Stores the average IQ of students who did not get placed."
    ),
    Variable(
        name="placement_crosstab",
        value=pd.DataFrame(),
        description="Stores cross-tabulation of placement by internship experience and academic performance level."
    ),
    Variable(
        name="placement_rates_by_internship_performance",
        value=pd.DataFrame(),
        description="Stores the placement rates DataFrame for each internship-performance combination in question 3."
    ),
    Variable(
        name="skills_stats_placed",
        value=pd.DataFrame(),
        description="Stores skills distribution statistics (named: 25%, 50%, 75%) for placed students by internship experience."
    ),
    Variable(
        name="competency_placement_rates",
        value={},
        description="Stores placement rates by competency level (Low, Medium, High)."
    ),
    Variable(
        name="competency_internship_placement_rates", 
        value=pd.DataFrame(),
        description="""Stores placement rates by competency level and internship experience.
        The DataFrame should have 'Competency_Level' as the index ('High', 'Medium', 'Low')
        and 'Internship_Experience' as columns ('No', 'Yes').
        The values should be the placement rates, represented as percentages from 0 to 100.
        """
    ),
    Variable(
        name="best_combination",
        value="",
        description="Stores the competency level and internship experience combination with highest placement rate, in the format of 'CompetencyLevel_InternshipExperience', for example 'High_Yes'."
    ),
    Variable(
        name="best_placement_rate",
        value=0.0,
        description="Stores the highest placement rate found in competency-internship combinations."
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