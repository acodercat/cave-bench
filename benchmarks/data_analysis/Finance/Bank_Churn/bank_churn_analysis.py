"""
This scenario is based on analyzing a Bank Customer Churn dataset.
The goal is to use pandas to analyze customer demographics, financial status, 
and their likelihood of exiting the bank.

Dataset Structure:
bank_churn.csv contains:
- RowNumber, CustomerId, Surname: Identifiers
- CreditScore: Numerical
- Geography: Categorical (France, Spain, Germany)
- Gender: Categorical
- Age, Tenure, Balance, NumOfProducts: Numerical
- HasCrCard, IsActiveMember, Exited: Binary (1 or 0)
- EstimatedSalary: Numerical

Question 1: Count the total number of customers in the dataset.
Question 2: Identify the number of unique geographical locations and their customer counts.
Question 3: Calculate the average Credit Score of female customers.
Question 4: Calculate the overall churn rate (mean of 'Exited') in percentage.
Question 5: Find the average Balance of customers who have exited (Exited=1).
Question 6: Count how many Female customers reside in France.
Question 7: Calculate the churn rate by Geography.
Question 8: Compare the average Estimated Salary of Active vs. Inactive members.
Question 9: Calculate the percentage of customers who have a Balance of 0.
Question 10: Identify the Surname that appears most frequently.
Question 11: Calculate the churn rate for customers with 3 or more products.
Question 12: Calculate the Pearson correlation between Age and CreditScore.
Question 13: Create Age groups (bins: 18-30, 31-40, 41-50, 51-60, 60+) and identify the group with the highest churn count.
Question 14: Create a Pivot Table showing mean 'Exited' rates with 'Geography' as index and 'Gender' as columns.
Question 15: Segment 'CreditScore' into 3 equal quantiles ('Low', 'Medium', 'High'). Find the (Geography, Credit_Segment) group with the highest churn rate.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd
import numpy as np

# Load the dataset
try:
    churn_df = pd.read_csv("benchmarks/data_analysis/Finance/Bank_Churn/Churn_Modelling.csv")
except Exception as e:
    print(f"Error loading Churn dataset: {e}")
    churn_df = None

tools = []

# Correct answers (Placeholder values based on typical Churn dataset properties)
correct_answers = {
    "q1": 10000,          # Total customers
    "q2": 3,              # Unique geographies (France, Spain, Germany)
    "q2_france": 5014,    # Number of customers in France
    "q2_spain": 2477,     # Number of customers in Spain
    "q2_germany": 2509,   # Number of customers in Germany
    "q3": 650.83,         # Mean Credit Score
    "q4": 20.37,          # Churn rate percentage
    "q5": 91108.54,       # Avg Balance of exited customers
    "q6": 2261,           # Count Female in France
    "q7_france": 16.15,   # Churn rate France
    "q7_germany": 32.44,  # Churn rate Germany
    "q7_spain": 16.67,    # Churn rate Spain
    "q8_active": 99452.97, # Avg Salary Active
    "q8_inactive": 100767.20, # Avg Salary Inactive
    "q9": 36.17,          # Percentage of zero balance
    "q10": "Smith",       # Most frequent surname
    "q11": 85.89,         # Churn rate for 3+ products (high churn usually)
    "q12": -0.0040,       # Correlation Age vs CreditScore
    "q13_bin": "41-50",   # Age group bin with highest churn count
    "q13_churn_count": 788, # Churn count for the age group
    "q14_fr_female": 20.34,  # Pivot value: France-Female churn rate
    "q14_sp_male": 13.11,    # Pivot value: Spain-Male churn rate
    "q15_group": ('Germany', 'Low'), # (Geography, Credit_Segment) group with the highest churn rate
    "q15_churn_rate": 33.65, # Churn rate for the (Geography, Credit_Segment) group
}

# --- Validators ---

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    total_customers = runtime.get_variable("total_customers")

    # Check type
    if not isinstance(total_customers, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q1: Expected int, got {type(total_customers).__name__}")
    
    # Check value
    if total_customers == correct_answers["q1"]:
        return ValidatorResult(success=True, message="Q1: Correct.")
    return ValidatorResult(success=False, message=f"Q1: Expected {correct_answers['q1']}, got {total_customers}")

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    unique_geos = runtime.get_variable("unique_geos")
    unique_geo_customer_counts = runtime.get_variable("unique_geo_customer_counts")

    # Check type
    if not isinstance(unique_geos, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q2: Expected int, got {type(unique_geos).__name__}")
    if not isinstance(unique_geo_customer_counts, pd.Series):
        return ValidatorResult(success=False, message=f"Q2: Expected pd.Series, got {type(unique_geo_customer_counts).__name__}")
    
    france_count = unique_geo_customer_counts.get("France", 0)
    spain_count = unique_geo_customer_counts.get("Spain", 0)
    germany_count = unique_geo_customer_counts.get("Germany", 0)
    unique_geos_correct = unique_geos == correct_answers["q2"]
    france_count_correct = france_count == correct_answers["q2_france"]
    spain_count_correct = spain_count == correct_answers["q2_spain"]
    germany_count_correct = germany_count == correct_answers["q2_germany"]

    if unique_geos_correct and france_count_correct and spain_count_correct and germany_count_correct:
        return ValidatorResult(success=True, message="Q2: Correct.")
    elif unique_geos_correct and not france_count_correct and not spain_count_correct and not germany_count_correct:
        return ValidatorResult(success=False, message=f"Q2: Incorrect values. Unique geos={unique_geos}, France={france_count}, Spain={spain_count}, Germany={germany_count}")
    elif unique_geos_correct and france_count_correct and not spain_count_correct and not germany_count_correct:
        return ValidatorResult(success=False, message=f"Q2: Incorrect values. Unique geos={unique_geos}, France={france_count}, Spain={spain_count}, Germany={germany_count}")
    elif unique_geos_correct and not france_count_correct and spain_count_correct and not germany_count_correct:
        return ValidatorResult(success=False, message=f"Q2: Incorrect values. Unique geos={unique_geos}, France={france_count}, Spain={spain_count}, Germany={germany_count}")
    elif unique_geos_correct and not france_count_correct and not spain_count_correct and germany_count_correct:
        return ValidatorResult(success=False, message=f"Q2: Incorrect values. Unique geos={unique_geos}, France={france_count}, Spain={spain_count}, Germany={germany_count}")
    return ValidatorResult(success=False, message=f"Q2: Incorrect values. Unique geos={unique_geos}, France={france_count}, Spain={spain_count}, Germany={germany_count}")

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    avg_credit_score_female = runtime.get_variable("avg_credit_score_female")

    # Check type
    if not isinstance(avg_credit_score_female, float):
        return ValidatorResult(success=False, message=f"Q3: Expected float, got {type(avg_credit_score_female).__name__}")
    
    # Check value
    if round(avg_credit_score_female, 2) == correct_answers["q3"]:
        return ValidatorResult(success=True, message="Q3: Correct.")
    return ValidatorResult(success=False, message=f"Q3: Expected {correct_answers['q3']}, got {round(avg_credit_score_female, 2)}")

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    churn_rate_pct = runtime.get_variable("churn_rate_pct")

    # Check type
    if not isinstance(churn_rate_pct, float):
        return ValidatorResult(success=False, message=f"Q4: Expected float, got {type(churn_rate_pct).__name__}")

    # Check value
    if round(churn_rate_pct, 2) == correct_answers["q4"]:
        return ValidatorResult(success=True, message="Q4: Correct.")
    return ValidatorResult(success=False, message=f"Q4: Expected {correct_answers['q4']}, got {round(churn_rate_pct, 2)}")

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    avg_balance_exited = runtime.get_variable("avg_balance_exited")

    # Check type
    if not isinstance(avg_balance_exited, float):
        return ValidatorResult(success=False, message=f"Q5: Expected float, got {type(avg_balance_exited).__name__}")
    
    # Check value
    if round(avg_balance_exited, 2) == correct_answers["q5"]:
        return ValidatorResult(success=True, message="Q5: Correct.")
    return ValidatorResult(success=False, message=f"Q5: Expected {correct_answers['q5']}, got {round(avg_balance_exited, 2)}")

def validate_q6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    female_france_count = runtime.get_variable("female_france_count")

    # Check type
    if not isinstance(female_france_count, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q6: Expected int, got {type(female_france_count).__name__}")

    # Check value
    if female_france_count == correct_answers["q6"]:
        return ValidatorResult(success=True, message="Q6: Correct.")
    return ValidatorResult(success=False, message=f"Q6: Expected {correct_answers['q6']}, got {female_france_count}")

def validate_q7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    geo_churn_rates = runtime.get_variable("geo_churn_rates")
    if not isinstance(geo_churn_rates, pd.Series):
        return ValidatorResult(success=False, message=f"Q7: Expected pd.Series, got {type(geo_churn_rates).__name__}")
    
    # Check specific values (e.g., France and Germany)
    fr_val = geo_churn_rates.get("France", 0)
    de_val = geo_churn_rates.get("Germany", 0)
    sp_val = geo_churn_rates.get("Spain", 0)
    
    if fr_val > 1:
        fr_val = round(fr_val, 2)
    else:
        fr_val = round(fr_val * 100, 2)
    if de_val > 1:
        de_val = round(de_val, 2)
    else:
        de_val = round(de_val * 100, 2)
    if sp_val > 1:
        sp_val = round(sp_val, 2)
    else:
        sp_val = round(sp_val * 100, 2)
    
    if fr_val == correct_answers["q7_france"] and de_val == correct_answers["q7_germany"] and sp_val == correct_answers["q7_spain"]:
        return ValidatorResult(success=True, message="Q7: Correct.")
    elif fr_val == correct_answers["q7_france"] and de_val == correct_answers["q7_germany"] and not sp_val == correct_answers["q7_spain"]:
        return ValidatorResult(success=False, message=f"Q7: Incorrect values. France={fr_val}, Germany={de_val}, Spain={sp_val}")
    elif fr_val == correct_answers["q7_france"] and not de_val == correct_answers["q7_germany"] and sp_val == correct_answers["q7_spain"]:
        return ValidatorResult(success=False, message=f"Q7: Incorrect values. France={fr_val}, Germany={de_val}, Spain={sp_val}")
    elif not fr_val == correct_answers["q7_france"] and de_val == correct_answers["q7_germany"] and sp_val == correct_answers["q7_spain"]:
        return ValidatorResult(success=False, message=f"Q7: Incorrect values. France={fr_val}, Germany={de_val}, Spain={sp_val}")
    elif not fr_val == correct_answers["q7_france"] and not de_val == correct_answers["q7_germany"] and sp_val == correct_answers["q7_spain"]:
        return ValidatorResult(success=False, message=f"Q7: Incorrect values. France={fr_val}, Germany={de_val}, Spain={sp_val}")
    elif not fr_val == correct_answers["q7_france"] and de_val == correct_answers["q7_germany"] and not sp_val == correct_answers["q7_spain"]:
        return ValidatorResult(success=False, message=f"Q7: Incorrect values. France={fr_val}, Germany={de_val}, Spain={sp_val}")
    elif fr_val == correct_answers["q7_france"] and not de_val == correct_answers["q7_germany"] and not sp_val == correct_answers["q7_spain"]:
        return ValidatorResult(success=False, message=f"Q7: Incorrect values. France={fr_val}, Germany={de_val}, Spain={sp_val}")
    return ValidatorResult(success=False, message=f"Q7: Incorrect values. France={fr_val}, Germany={de_val}, Spain={sp_val}")

def validate_q8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    active_inactive_salary = runtime.get_variable("active_inactive_salary")

    # Check type
    if not isinstance(active_inactive_salary, pd.Series):
        return ValidatorResult(success=False, message=f"Q8: Expected pd.Series, got {type(active_inactive_salary).__name__}")
    
    # Handle both cases: index as 'Active'/'Inactive' or as 0/1
    if 'Active' in active_inactive_salary.index or 'Inactive' in active_inactive_salary.index:
        active_salary = round(active_inactive_salary.get('Active', 0), 2)
        inactive_salary = round(active_inactive_salary.get('Inactive', 0), 2)
    else:
        # Index is 0/1: 1 = Active, 0 = Inactive
        active_salary = round(active_inactive_salary.get(1, 0), 2)
        inactive_salary = round(active_inactive_salary.get(0, 0), 2)

    # Check value
    active_salary_correct = active_salary == correct_answers["q8_active"]
    inactive_salary_correct = inactive_salary == correct_answers["q8_inactive"]
    if active_salary_correct and inactive_salary_correct:
        return ValidatorResult(success=True, message="Q8: Correct.")
    elif active_salary_correct and not inactive_salary_correct:
        return ValidatorResult(success=False, message=f"Q8: Incorrect values. Active={active_salary}, Inactive={inactive_salary}")
    elif not active_salary_correct and inactive_salary_correct:
        return ValidatorResult(success=False, message=f"Q8: Incorrect values. Active={active_salary}, Inactive={inactive_salary}")
    return ValidatorResult(success=False, message=f"Q8: Incorrect values. Active={active_salary}, Inactive={inactive_salary}")

def validate_q9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    zero_balance_pct = runtime.get_variable("zero_balance_pct")

    # Check type
    if not isinstance(zero_balance_pct, float):
        return ValidatorResult(success=False, message=f"Q9: Expected float, got {type(zero_balance_pct).__name__}")
    
    # Check value
    if round(zero_balance_pct, 2) == correct_answers["q9"]:
        return ValidatorResult(success=True, message="Q9: Correct.")
    return ValidatorResult(success=False, message=f"Q9: Expected {correct_answers['q9']}, got {round(zero_balance_pct, 2)}")

def validate_q10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    top_surname = runtime.get_variable("top_surname")

    # Check type
    if not isinstance(top_surname, str):
        return ValidatorResult(success=False, message=f"Q10: Expected str, got {type(top_surname).__name__}")
    
    # Check value
    if top_surname == correct_answers["q10"]:
        return ValidatorResult(success=True, message="Q10: Correct.")
    return ValidatorResult(success=False, message=f"Q10: Expected {correct_answers['q10']}, got {top_surname}")

def validate_q11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    multi_product_churn_rate = runtime.get_variable("multi_product_churn_rate")

    # Check type
    if not isinstance(multi_product_churn_rate, float):
        return ValidatorResult(success=False, message=f"Q11: Expected float, got {type(multi_product_churn_rate).__name__}")
    
    if multi_product_churn_rate > 1:
        normalized_multi_product_churn_rate = round(multi_product_churn_rate, 2)
    else:
        normalized_multi_product_churn_rate = round(multi_product_churn_rate * 100, 2)

    # Check value
    if normalized_multi_product_churn_rate == correct_answers["q11"]:
        return ValidatorResult(success=True, message="Q11: Correct.")
    return ValidatorResult(success=False, message=f"Q11: Incorrect. Expected {correct_answers['q11']}, got {normalized_multi_product_churn_rate}")

def validate_q12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    age_credit_corr = runtime.get_variable("age_credit_corr")

    # Check type
    if not isinstance(age_credit_corr, float):
        return ValidatorResult(success=False, message=f"Q12: Expected float, got {type(age_credit_corr).__name__}")

    # Check value
    if round(age_credit_corr, 4) == correct_answers["q12"]:
        return ValidatorResult(success=True, message="Q12: Correct.")
    return ValidatorResult(success=False, message=f"Q12: Incorrect. Expected {correct_answers['q12']}, got {round(age_credit_corr, 4)}")

def validate_q13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    risky_age_group = runtime.get_variable("risky_age_group")
    risky_age_group_churn_count = runtime.get_variable("risky_age_group_churn_count")

    # Check types
    if not isinstance(risky_age_group, str):
        return ValidatorResult(success=False, message=f"Q13: Expected str, got {type(risky_age_group).__name__}")
    if not isinstance(risky_age_group_churn_count, (int, np.integer)):
        return ValidatorResult(success=False, message=f"Q13: Expected int, got {type(risky_age_group_churn_count).__name__}")
    
    # Check values
    group_correct = risky_age_group == correct_answers["q13_bin"]
    churn_count_correct = risky_age_group_churn_count == correct_answers["q13_churn_count"]
    if group_correct and churn_count_correct:
        return ValidatorResult(success=True, message="Q13: Correct.")
    elif group_correct and not churn_count_correct:
        return ValidatorResult(success=False, message=f"Q13: Incorrect. Expected {correct_answers['q13_churn_count']}, got {risky_age_group_churn_count}")
    elif not group_correct and churn_count_correct:
        return ValidatorResult(success=False, message=f"Q13: Incorrect. Expected {correct_answers['q13_bin']}, got {risky_age_group}")
    return ValidatorResult(success=False, message=f"Q13: Incorrect. Expected {correct_answers['q13_bin']}, got {risky_age_group} and {correct_answers['q13_churn_count']}, got {risky_age_group_churn_count}")

def validate_q14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    geo_gender_pivot = runtime.get_variable("geo_gender_pivot") 

    # Check type
    if not isinstance(geo_gender_pivot, pd.DataFrame):
        return ValidatorResult(success=False, message=f"Q14: Expected DataFrame, got {type(geo_gender_pivot).__name__}")
    
    # Check shape and a specific value
    try:
        fr_fem_val = geo_gender_pivot['Female'].get('France')
        sp_male_val = geo_gender_pivot['Male'].get('Spain')
    except:
        return ValidatorResult(success=False, message="Q14: Error accessing France Female and Spain Male in pivot table.")

    if fr_fem_val > 1:
        fr_fem_val = round(fr_fem_val, 2)
    else:
        fr_fem_val = round(fr_fem_val * 100, 2)
    
    if sp_male_val > 1:
        sp_male_val = round(sp_male_val, 2)
    else:
        sp_male_val = round(sp_male_val * 100, 2)

    fr_fem_correct = fr_fem_val == correct_answers["q14_fr_female"]
    sp_male_correct = sp_male_val == correct_answers["q14_sp_male"]
    if fr_fem_correct and sp_male_correct:
        return ValidatorResult(success=True, message="Q14: Correct.")
    elif fr_fem_correct and not sp_male_correct:
        return ValidatorResult(success=False, message=f"Q14: Incorrect. Expected {correct_answers['q14_sp_male']}, got {sp_male_val}")
    elif not fr_fem_correct and sp_male_correct:
        return ValidatorResult(success=False, message=f"Q14: Incorrect. Expected {correct_answers['q14_fr_female']}, got {fr_fem_val}")
    return ValidatorResult(success=False, message=f"Q14: Incorrect. Expected {correct_answers['q14_fr_female']}, got {fr_fem_val} and {correct_answers['q14_sp_male']}, got {sp_male_val}")
    
def validate_q15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    group_with_highest_churn_rate = runtime.get_variable("group_with_highest_churn_rate")
    churn_rate_for_group = runtime.get_variable("churn_rate_for_group")

    # Check type
    if not isinstance(group_with_highest_churn_rate, tuple):
        return ValidatorResult(success=False, message=f"Q15: Expected tuple, got {type(group_with_highest_churn_rate).__name__}")
    if not isinstance(churn_rate_for_group, float):
        return ValidatorResult(success=False, message=f"Q15: Expected float, got {type(churn_rate_for_group).__name__}")
    
    if churn_rate_for_group > 1:
        churn_rate_for_group = round(churn_rate_for_group, 2)
    else:
        churn_rate_for_group = round(churn_rate_for_group * 100, 2)
        
    # Check value
    group_correct = group_with_highest_churn_rate == correct_answers["q15_group"]
    churn_rate_correct = churn_rate_for_group == correct_answers["q15_churn_rate"]
    if group_correct and churn_rate_correct:
        return ValidatorResult(success=True, message="Q15: Correct.")
    elif group_correct and not churn_rate_correct:
        return ValidatorResult(success=False, message=f"Q15: Incorrect. Expected {correct_answers['q15_churn_rate']}, got {churn_rate_for_group}")
    elif not group_correct and churn_rate_correct:
        return ValidatorResult(success=False, message=f"Q15: Incorrect. Expected {correct_answers['q15_group']}, got {group_with_highest_churn_rate}")
    return ValidatorResult(success=False, message=f"Q15: Incorrect. Expected {correct_answers['q15_group']}, got {group_with_highest_churn_rate} and {correct_answers['q15_churn_rate']}, got {churn_rate_for_group}")


variables = [
    # Dataset
    Variable(
        name="churn_df",
        value=churn_df,
        description="""
        A pandas DataFrame loaded from Churn_Modelling.csv.
        Columns and their dtypes:
        - RowNumber: int64
        - CustomerId: int64
        - Surname: object (string)
        - CreditScore: int64
        - Geography: object (string, 'France', 'Spain', 'Germany')
        - Gender: object (string, 'Female', 'Male')
        - Age: int64
        - Tenure: int64
        - Balance: float64
        - NumOfProducts: int64
        - HasCrCard: int64 (0 or 1)
        - IsActiveMember: int64 (0 or 1)
        - EstimatedSalary: float64
        - Exited: int64 (0 or 1)
        Example: pd.DataFrame(data={
            "RowNumber": [1],
            "CustomerId": [15634602],
            "Surname": ["Hargrave"],
            "CreditScore": [619],
            "Geography": ["France"],
            "Gender": ["Female"],
            "Age": [42],
            "Tenure": [2],
            "Balance": [0.0],   
            "NumOfProducts": [1],
            "HasCrCard": [1],
            "IsActiveMember": [1],
            "EstimatedSalary": [101348.88],
            "Exited": [1]
        })
        You should use this variable to answer all questions.
        """
    ),
    # Q1
    Variable(
        name="total_customers",
        value=0,
        description="Int. Store the total number of customers in the dataset of question 1.",
    ),
    # Q2
    Variable(
        name="unique_geos",
        value=0,
        description="Int. Store the number of unique entries in 'Geography' column of question 2.",
    ),
    Variable(
        name="unique_geo_customer_counts",
        value=pd.Series(dtype=int),
        description="Series. Store the count of customers by uniqueGeography of question 2.",
    ),
    # Q3
    Variable(
        name="avg_credit_score_female",
        value=0.0,
        description="Float. Store the mean of 'CreditScore' for female customers of question 3.",
    ),
    # Q4
    Variable(
        name="churn_rate_pct",
        value=0.0,
        description="Float. Store the percentage of customers where Exited=1 of question 4.",
    ),
    # Q5
    Variable(
        name="avg_balance_exited",
        value=0.0,
        description="Float. Store the mean 'Balance' of customers who Exited of question 5.",
    ),
    # Q6
    Variable(
        name="female_france_count",
        value=0,
        description="Int. Store the count of customers who are Female AND in France of question 6.",
    ),
    # Q7
    Variable(
        name="geo_churn_rates", 
        value=pd.Series(dtype=float), 
        description="Series. Store the churn rate grouped by Geography of question 7."),
    # Q8
    Variable(
        name="active_inactive_salary",
        value=pd.Series(dtype=float),
        description="Series. Store the mean EstimatedSalary grouped by IsActiveMember (Active or Inactive) of question 8.",
    ),
    # Q9
    Variable(
        name="zero_balance_pct", 
        value=0.0, 
        description="Float. Store the percentage of customers with Balance == 0 of question 9.",
    ),
    # Q10
    Variable(
        name="top_surname", 
        value="", 
        description="String. Store the most frequent Surname of question 10.",
    ),
    # Q11
    Variable(
        name="multi_product_churn_rate", 
        value=0.0, 
        description="Float. Store the churn rate for customers with NumOfProducts >= 3 of question 11."
    ),
    # Q12
    Variable(
        name="age_credit_corr", 
        value=0.0, 
        description="Float. Store the correlation between Age and CreditScore of question 12."
    ),
    # Q13
    Variable(
        name="risky_age_group", 
        value="", 
        description="String. Store the age bin (e.g., '41-50') with highest churn count of question 13."
    ),
    Variable(
        name="risky_age_group_churn_count",
        value=0,
        description="Int. Store the churn count for the age bin with highest churn count of question 13."
    ),
    # Q14
    Variable(
        name="geo_gender_pivot", 
        value=pd.DataFrame(), 
        description="DataFrame. Store the pivot table of mean Exited (Index: Geography, Col: Gender) of question 14."),
    # Q15
    Variable(
        name="group_with_highest_churn_rate", 
        value=(), 
        description="Tuple. Store the (Geography, Credit_Segment) group with the highest churn rate of question 15."
    ),
    Variable(
        name="churn_rate_for_group",
        value=0.0,
        description="Float. Store the churn rate for the (Geography, Credit_Segment) group of question 15."
    )
]

validators = {
    "validate_q1": validate_q1,
    "validate_q2": validate_q2,
    "validate_q3": validate_q3,
    "validate_q4": validate_q4,
    "validate_q5": validate_q5,
    "validate_q6": validate_q6,
    "validate_q7": validate_q7,
    "validate_q8": validate_q8,
    "validate_q9": validate_q9,
    "validate_q10": validate_q10,
    "validate_q11": validate_q11,
    "validate_q12": validate_q12,
    "validate_q13": validate_q13,
    "validate_q14": validate_q14,
    "validate_q15": validate_q15,
}

if __name__ == "__main__":
    pass