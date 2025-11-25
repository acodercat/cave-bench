"""
This scenario is based on analyzing the Heart Attack Risk Prediction Dataset.
The goal is to use pandas and data analysis techniques to progressively analyze
heart disease risk factors and patterns.

Dataset Structure:
heart_attack.csv contains the following columns:
- Patient ID: unique identifier for each patient
- Age: Age of the patient
- Sex: Gender (Female, Male)
- Cholesterol: Cholesterol level (mg/dL)
- Blood Pressure: Systolic/Diastolic BP (format: "120/80")
- Heart Rate: Heart rate (bpm)
- Diabetes: Diabetes status (0 = No, 1 = Yes)
- Family History: Family history of heart disease (0 = No, 1 = Yes)
- Smoking: Smoking status (0 = No, 1 = Yes)
- Obesity: Obesity status (0 = No, 1 = Yes)
- Alcohol Consumption: Alcohol consumption (0 = No, 1 = Yes)
- Exercise Hours Per Week: Weekly exercise hours
- Diet: Diet quality (Average, Healthy, Unhealthy)
- Previous Heart Problems: Previous heart issues (0 = No, 1 = Yes)
- Medication Use: Medication use (0 = No, 1 = Yes)
- Stress Level: Stress level (1-10 scale)
- Sedentary Hours Per Day: Daily sedentary hours
- Income: Annual income
- BMI: Body Mass Index
- Triglycerides: Triglyceride levels (mg/dL)
- Physical Activity Days Per Week: Days of physical activity per week
- Sleep Hours Per Day: Daily sleep hours
- Country: Country of residence
- Continent: Continent
- Hemisphere: Hemisphere
- Heart Attack Risk: Risk of heart attack (0 = No, 1 = Yes)

Question 1: Calculate heart attack risk rate by age groups
Divide patients into age groups (Under 40, 40-50, 50-60, 60+) and calculate 
the heart attack risk rate for each group. Store as a pandas Series.

Question 2: Find average cholesterol and BMI for high-risk vs low-risk patients
Calculate the average cholesterol level and BMI for patients with heart attack risk
versus those without. Handle any data type conversions appropriately.

Question 3: Analyze risk patterns by lifestyle factors
Create a comprehensive analysis of heart attack risk based on lifestyle factors
(Smoking, Obesity, Exercise). Calculate risk rates for each combination and identify
the most dangerous lifestyle pattern.

Question 4: Analyze blood pressure patterns and stress correlation
Extract systolic and diastolic BP from the Blood Pressure column, then analyze:
- Average BP values for high-risk patients by diabetes status
- Correlation between stress level and systolic BP for high-risk patients

Question 5: Complex multi-factor risk analysis
Perform comprehensive risk analysis:
- Create risk score categories based on number of risk factors (Diabetes, Smoking, Obesity, 
  Family History, Previous Heart Problems)
- Calculate heart attack rates by risk score and diet quality
- Identify the most protective combination of exercise hours and diet
- Find the threshold values for cholesterol and triglycerides that best separate 
  high-risk from low-risk patients
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd

# Load the Heart Attack Risk Prediction dataset
heart_df = pd.read_csv("benchmarks/data_analysis/HeartAttack/heart_attack_prediction_dataset.csv")

tools = []

# Correct answers (calculated from actual dataset)
correct_answers = {
    "q1_under40": 0.3579,  # Risk rate for Under 40
    "q1_40to50": 0.3672,   # Risk rate for 40-50
    "q1_50to60": 0.3400,   # Risk rate for 50-60  
    "q1_60plus": 0.3612,   # Risk rate for 60+
    "q2_cholesterol_high_risk": 261.97,  # Avg cholesterol for high-risk
    "q2_cholesterol_low_risk": 258.71,   # Avg cholesterol for low-risk
    "q2_bmi_high_risk": 28.89,           # Avg BMI for high-risk
    "q2_bmi_low_risk": 28.89,            # Avg BMI for low-risk
    "q3_no_smoking_obesity_low_exercise": 0.5000,  # Risk rate for worst lifestyle
    "q3_no_smoking_obesity_high_exercise": 0.3100,  # Risk rate for best lifestyle
    "q4_systolic_diabetes": 135.5,       # Avg systolic BP for diabetic high-risk
    "q4_systolic_no_diabetes": 136.1,    # Avg systolic BP for non-diabetic high-risk
    "q4_stress_bp_correlation": 0.013,  # Correlation between stress and systolic BP
    "q5_high_risk_category": 0.3632,     # Risk rate for 4-5 risk factors
    "q5_best_exercise_diet": "Medium_Unhealthy",  # Best exercise-diet combination
    "q5_cholesterol_threshold": 330,     # Optimal cholesterol threshold
    "q5_triglycerides_threshold": 612    # Optimal triglycerides threshold
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the heart attack risk rate by age groups.
    """
    risk_by_age_group = runtime.get_variable_value("risk_by_age_group")
    
    if risk_by_age_group is None:
        return ValidatorResult(
            success=False,
            message="Variable 'risk_by_age_group' not found."
        )
    
    if not isinstance(risk_by_age_group, pd.Series):
        return ValidatorResult(
            success=False,
            message="Expected a pandas Series for risk_by_age_group."
        )
    
    # Handle percentage values
    risk_by_age_group = risk_by_age_group.copy()
    if (risk_by_age_group.values > 1).any():
        risk_by_age_group = risk_by_age_group / 100.0
    
    tolerance = 0.02
    checks = [
        ("Under 40", correct_answers["q1_under40"]),
        ("40-50", correct_answers["q1_40to50"]),
        ("50-60", correct_answers["q1_50to60"]),
        ("60+", correct_answers["q1_60plus"])
    ]
    
    for age_group, expected in checks:
        if age_group not in risk_by_age_group.index:
            return ValidatorResult(
                success=False,
                message=f"Missing age group '{age_group}' in risk_by_age_group."
            )
        
        actual = risk_by_age_group[age_group]
        if abs(actual - expected) > tolerance:
            return ValidatorResult(
                success=False,
                message=f"Incorrect risk rate for {age_group}. Expected ~{expected:.4f}, got {actual:.4f}."
            )
    
    return ValidatorResult(
        success=True,
        message="Correct. Heart attack risk rates by age group calculated successfully."
    )

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates average cholesterol and BMI for high-risk vs low-risk patients.
    """
    avg_cholesterol_high_risk = runtime.get_variable_value("avg_cholesterol_high_risk")
    avg_cholesterol_low_risk = runtime.get_variable_value("avg_cholesterol_low_risk")
    avg_bmi_high_risk = runtime.get_variable_value("avg_bmi_high_risk")
    avg_bmi_low_risk = runtime.get_variable_value("avg_bmi_low_risk")
    
    if any(var is None for var in [avg_cholesterol_high_risk, avg_cholesterol_low_risk, 
                                    avg_bmi_high_risk, avg_bmi_low_risk]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for cholesterol/BMI analysis."
        )
    
    tolerance = 2.0
    
    chol_high_correct = abs(avg_cholesterol_high_risk - correct_answers["q2_cholesterol_high_risk"]) <= tolerance
    chol_low_correct = abs(avg_cholesterol_low_risk - correct_answers["q2_cholesterol_low_risk"]) <= tolerance
    bmi_high_correct = abs(avg_bmi_high_risk - correct_answers["q2_bmi_high_risk"]) <= tolerance
    bmi_low_correct = abs(avg_bmi_low_risk - correct_answers["q2_bmi_low_risk"]) <= tolerance
    
    if all([chol_high_correct, chol_low_correct, bmi_high_correct, bmi_low_correct]):
        return ValidatorResult(
            success=True,
            message=f"Correct. Cholesterol - High Risk: {avg_cholesterol_high_risk:.2f}, Low Risk: {avg_cholesterol_low_risk:.2f}. BMI - High Risk: {avg_bmi_high_risk:.2f}, Low Risk: {avg_bmi_low_risk:.2f}."
        )
    else:
        return ValidatorResult(
            success=False,
            message="Incorrect averages for cholesterol or BMI by risk group."
        )

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates lifestyle factors risk analysis.
    """
    lifestyle_risk_rates = runtime.get_variable_value("lifestyle_risk_rates")
    most_dangerous_pattern = runtime.get_variable_value("most_dangerous_pattern")
    safest_pattern = runtime.get_variable_value("safest_pattern")
    
    if lifestyle_risk_rates is None:
        return ValidatorResult(
            success=False,
            message="Variable 'lifestyle_risk_rates' not found."
        )
    
    tolerance = 0.03
    
    try:
        # Check worst lifestyle combination (Smoking=0, Obesity=1, Low Exercise)
        if isinstance(lifestyle_risk_rates, pd.DataFrame) or isinstance(lifestyle_risk_rates, pd.Series):
            worst_rate = lifestyle_risk_rates.loc[(0, 1, 'Low')]
            best_rate = lifestyle_risk_rates.loc[(0, 1, 'High')]
            # Convert to scalar if Series
            if isinstance(worst_rate, pd.Series):
                worst_rate = float(worst_rate.iloc[0])
            else:
                worst_rate = float(worst_rate)
            if isinstance(best_rate, pd.Series):
                best_rate = float(best_rate.iloc[0])
            else:
                best_rate = float(best_rate)
        else:
            return ValidatorResult(
                success=False,
                message="lifestyle_risk_rates should be a pandas DataFrame or Series."
            )

        if worst_rate > 1:
            worst_rate = worst_rate / 100.0
        if best_rate > 1:
            best_rate = best_rate / 100.0
            
        worst_correct = abs(worst_rate - correct_answers["q3_no_smoking_obesity_low_exercise"]) <= tolerance
        best_correct = abs(best_rate - correct_answers["q3_no_smoking_obesity_high_exercise"]) <= tolerance
        
        if worst_correct and best_correct:
            return ValidatorResult(
                success=True,
                message=f"Correct. Identified most dangerous pattern: {most_dangerous_pattern} and safest pattern: {safest_pattern}."
            )
        else:
            return ValidatorResult(
                success=False,
                message="Incorrect risk rates for lifestyle patterns."
            )
    except (KeyError, IndexError, AttributeError):
        return ValidatorResult(
            success=False,
            message="Error accessing lifestyle risk rates. Check the structure of lifestyle_risk_rates."
        )

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates blood pressure and stress correlation analysis.
    """
    avg_systolic_diabetes = runtime.get_variable_value("avg_systolic_diabetes")
    avg_systolic_no_diabetes = runtime.get_variable_value("avg_systolic_no_diabetes")
    stress_bp_correlation = runtime.get_variable_value("stress_bp_correlation")
    
    if any(var is None for var in [avg_systolic_diabetes, avg_systolic_no_diabetes, stress_bp_correlation]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for BP analysis."
        )
    
    tolerance_bp = 3.0
    tolerance_corr = 0.05
    
    diabetes_correct = abs(avg_systolic_diabetes - correct_answers["q4_systolic_diabetes"]) <= tolerance_bp
    no_diabetes_correct = abs(avg_systolic_no_diabetes - correct_answers["q4_systolic_no_diabetes"]) <= tolerance_bp
    correlation_correct = abs(stress_bp_correlation - correct_answers["q4_stress_bp_correlation"]) <= tolerance_corr
    
    if all([diabetes_correct, no_diabetes_correct, correlation_correct]):
        return ValidatorResult(
            success=True,
            message=f"Correct. BP analysis completed. Stress-BP correlation: {stress_bp_correlation:.4f}."
        )
    else:
        return ValidatorResult(
            success=False,
            message="Incorrect blood pressure or correlation values."
        )

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates complex multi-factor risk analysis.
    """
    risk_score_rates = runtime.get_variable_value("risk_score_rates")
    best_exercise_diet_combo = runtime.get_variable_value("best_exercise_diet_combo")
    print(type(best_exercise_diet_combo))
    cholesterol_threshold = runtime.get_variable_value("cholesterol_threshold")
    triglycerides_threshold = runtime.get_variable_value("triglycerides_threshold")
    
    if any(var is None for var in [risk_score_rates, best_exercise_diet_combo, 
                                   cholesterol_threshold, triglycerides_threshold]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for multi-factor analysis."
        )
    
    tolerance_rate = 0.03
    tolerance_threshold = 10
    
    try:
        # Check high risk category rate (4-5 risk factors)
        if isinstance(risk_score_rates, pd.Series):
            high_risk_rate = risk_score_rates.loc['High']
        else:
            high_risk_rate = risk_score_rates['High']
            
        if high_risk_rate > 1:
            high_risk_rate = high_risk_rate / 100.0
            
        rate_correct = abs(high_risk_rate - correct_answers["q5_high_risk_category"]) <= tolerance_rate
        # Convert tuple to string format if it's a tuple
        if isinstance(best_exercise_diet_combo, tuple) and len(best_exercise_diet_combo) == 2:
            combo_str = "_".join(best_exercise_diet_combo)
        else:
            combo_str = str(best_exercise_diet_combo).strip()
        combo_correct = combo_str == correct_answers["q5_best_exercise_diet"]
        chol_correct = abs(cholesterol_threshold - correct_answers["q5_cholesterol_threshold"]) <= tolerance_threshold
        trig_correct = abs(triglycerides_threshold - correct_answers["q5_triglycerides_threshold"]) <= tolerance_threshold
        
        if all([rate_correct, combo_correct, chol_correct, trig_correct]):
            return ValidatorResult(
                success=True,
                message=f"Correct. Complex analysis completed. Best combo: {best_exercise_diet_combo}, Thresholds: Chol={cholesterol_threshold}, Trig={triglycerides_threshold}."
            )
        else:
            error_parts = []
            if not rate_correct:
                error_parts.append(f"High risk rate incorrect")
            if not combo_correct:
                error_parts.append(f"Best combo expected {correct_answers['q5_best_exercise_diet']}, got {combo_str}")
            if not chol_correct:
                error_parts.append(f"Cholesterol threshold incorrect")
            if not trig_correct:
                error_parts.append(f"Triglycerides threshold incorrect")
            
            return ValidatorResult(
                success=False,
                message=f"Incorrect multi-factor analysis: {'; '.join(error_parts)}."
            )
    except (KeyError, IndexError, AttributeError) as e:
        return ValidatorResult(
            success=False,
            message=f"Error in multi-factor analysis validation: {str(e)}"
        )

variables = [
    Variable(
        name="heart_df",
        value=heart_df,
        description="""A pandas DataFrame from 'heart_attack_prediction_dataset.csv' containing heart disease risk data.
        Columns and their dtypes: 
        - Patient ID: object, 
        - Age: int64, 
        - Sex: object, 
        - Cholesterol: int64, 
        - Blood Pressure: object, 
        - Heart Rate: int64, 
        - Diabetes: int64, 
        - Family History: int64,
        - Smoking: int64, 
        - Obesity: int64, 
        - Alcohol Consumption: int64, 
        - Exercise Hours Per Week: float64, 
        - Diet: object, 
        - Previous Heart Problems: int64,
        - Medication Use: int64, 
        - Stress Level: int64, 
        - Sedentary Hours Per Day: float64, 
        - Income: int64, 
        - BMI: float64, 
        - Triglycerides: int64,
        - Physical Activity Days Per Week: int64, 
        - Sleep Hours Per Day: int64, 
        - Country: object, 
        - Continent: object, 
        - Hemisphere: object,
        - Heart Attack Risk: int64.

        Example: pd.DataFrame(data={
            "Patient ID": ["BMW7812"],
            "Age": [67],
            "Sex": ["Male"],
            "Cholesterol": [208],
            "Blood Pressure": ["158/88"],
            "Heart Rate": [72],
            "Diabetes": [0],
            "Family History": [0],
            "Smoking": [1],
            "Obesity": [0],
            "Alcohol Consumption": [0],
            "Exercise Hours Per Week": [4.168189],
            "Diet": ["Average"],
            "Previous Heart Problems": [0],
            "Medication Use": [0],
            "Stress Level": [9],
            "Sedentary Hours Per Day": [6.615001],
            "Income": [261404],
            "BMI": [31.251233],
            "Triglycerides": [286],
            "Physical Activity Days Per Week": [0],
            "Sleep Hours Per Day": [6],
            "Country": ["Argentina"],
            "Continent": ["South America"],
            "Hemisphere": ["Southern Hemisphere"],
            "Heart Attack Risk": [0]
        })

        You should use this variable to answer all questions.
        """
    ),
    Variable(
        name="risk_by_age_group",
        value=pd.Series(dtype=float),
        description="Stores heart attack risk rate for each age group as a pandas Series."
    ),
    Variable(
        name="avg_cholesterol_high_risk",
        value=0.0,
        description="Stores average cholesterol level for high-risk patients."
    ),
    Variable(
        name="avg_cholesterol_low_risk",
        value=0.0,
        description="Stores average cholesterol level for low-risk patients."
    ),
    Variable(
        name="avg_bmi_high_risk",
        value=0.0,
        description="Stores average BMI for high-risk patients."
    ),
    Variable(
        name="avg_bmi_low_risk",
        value=0.0,
        description="Stores average BMI for low-risk patients."
    ),
    Variable(
        name="lifestyle_risk_rates",
        value=pd.DataFrame(),
        description="""Stores risk rates for different lifestyle factor combinations.
        Must be a pandas DataFrame or Series with MultiIndex (Smoking, Obesity, Exercise).
        - Smoking: 0 (No) or 1 (Yes)
        - Obesity: 0 (No) or 1 (Yes)  
        - Exercise: 'Low', 'Medium', or 'High' (string values)
        
        Example structure:
        MultiIndex [(0, 0, 'High'), (0, 0, 'Medium'), (0, 0, 'Low'), 
                   (0, 1, 'High'), (0, 1, 'Medium'), (0, 1, 'Low'),
                   (1, 0, 'High'), (1, 0, 'Medium'), (1, 0, 'Low'),
                   (1, 1, 'High'), (1, 1, 'Medium'), (1, 1, 'Low')]
        
        Values should be the risk rate (as decimal or percentage) for each combination.
        Access pattern: lifestyle_risk_rates.loc[(smoking_value, obesity_value, exercise_level)]"""
    ),
    Variable(
        name="most_dangerous_pattern",
        value="",
        description="""Stores the most dangerous lifestyle pattern as a tuple: (Smoking, Obesity, Exercise).
        Example: (1, 1, 'Low') representing Smoking=Yes, Obesity=Yes, Exercise=Low"""
    ),
    Variable(
        name="safest_pattern",
        value="",
        description="""Stores the safest lifestyle pattern as a tuple: (Smoking, Obesity, Exercise).
        Example: (0, 0, 'High') representing Smoking=No, Obesity=No, Exercise=High"""
    ),
    Variable(
        name="avg_systolic_diabetes",
        value=0.0,
        description="Stores average systolic BP for high-risk diabetic patients."
    ),
    Variable(
        name="avg_systolic_no_diabetes",
        value=0.0,
        description="Stores average systolic BP for high-risk non-diabetic patients."
    ),
    Variable(
        name="stress_bp_correlation",
        value=0.0,
        description="Stores correlation between stress level and systolic BP for high-risk patients."
    ),
    Variable(
        name="risk_score_rates",
        value=pd.Series(dtype=float),
        description="Stores heart attack rates by risk score category."
    ),
    Variable(
        name="risk_diet_analysis",
        value=pd.DataFrame(),
        description="Stores risk analysis by risk score and diet quality."
    ),
    Variable(
        name="best_exercise_diet_combo",
        value="",
        description="""Stores the best exercise-diet combination for heart health as a tuple.
        Format: (exercise_level, diet_quality)
        Example: ('Medium', 'Unhealthy') or ('High', 'Healthy')
        Exercise levels: 'Low', 'Medium', 'High'
        Diet quality: 'Healthy', 'Average', 'Unhealthy'"""
    ),
    Variable(
        name="cholesterol_threshold",
        value=0.0,
        description="Stores the 75th percentile of cholesterol levels as threshold."
    ),
    Variable(
        name="triglycerides_threshold",
        value=0.0,
        description="Stores the 75th percentile of triglycerides levels as threshold."
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