"""
This scenario is based on analyzing the famous Iris dataset from scikit-learn.
The goal is to use pandas and data analysis techniques to progressively analyze
the characteristics of different iris species and their distinguishing features.

Dataset Structure:
iris.csv contains the following columns:
- sepal_length: Sepal length in cm
- sepal_width: Sepal width in cm
- petal_length: Petal length in cm
- petal_width: Petal width in cm
- species: Species name (setosa, versicolor, virginica)

Question 1: Calculate mean petal length by species
Calculate the mean petal length for each iris species.
Store the result as a pandas Series with species as index and mean petal length as values.

Question 2: Find samples with extreme sepal lengths
Identify the samples with the maximum and minimum sepal lengths in the entire dataset.
Calculate the difference in all features between these two extreme samples.

Question 3: Analyze feature correlations
Calculate the correlation matrix for all numerical features (excluding species).
Identify the pair of features with the highest positive correlation and store their names.

Question 4: Comprehensive statistical analysis by species
For each species, calculate comprehensive statistics (mean, std, min, 25%, 50%, 75%, max) 
for all numerical features. Then identify which feature has the highest coefficient of 
variation (std/mean) across species to determine the most discriminative feature.

Question 5: Advanced shape analysis
Create new features representing flower shape characteristics:
- Petal shape ratio (petal_length / petal_width)
- Sepal shape ratio (sepal_length / sepal_width)
- Size ratio (petal_length / sepal_length)
Then analyze which combination of original and derived features best separates the species
by calculating the between-species variance to within-species variance ratio.
"""

from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn
import pandas as pd
from typing import List
from core.types import ToolCall

# Load the Iris dataset
iris_df = pd.read_csv("benchmarks/data_analysis/Iris/iris.csv")
iris_df['Species'] = iris_df['Species'].str.replace('Iris-', '')
iris_df.columns = ['id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

tools = []

# Correct answers (calculated from actual Iris dataset)
correct_answers = {
    "q1_setosa": 1.462,  # Mean petal length for setosa
    "q1_versicolor": 4.260,  # Mean petal length for versicolor
    "q1_virginica": 5.552,  # Mean petal length for virginica
    "q2_max_sepal_idx": 131,  # Index of sample with max sepal length
    "q2_min_sepal_idx": 13,  # Index of sample with min sepal length
    "q2_sepal_length_diff": 3.6,  # Difference in sepal length
    "q2_petal_length_diff": 5.4,  # Difference in petal length
    "q3_highest_corr_pair": ["petal_length", "petal_width"],  # Most correlated feature pair
    "q3_correlation_value": 0.9629,  # Correlation value
    "q4_most_variable_feature": "petal_width",  # Feature with highest CV across species
    "q4_setosa_petal_width_std": 0.1054,  # Std of petal width for setosa
    "q4_virginica_sepal_length_mean": 6.588,  # Mean sepal length for virginica
    "q5_best_shape_feature": "size_ratio",  # Best shape feature for separation
    "q5_best_f_ratio": 28.3234,  # F-ratio for best feature
    "q5_best_combination": ["petal_length", "petal_width", "size_ratio"],  # Best feature combination
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the mean petal length by species.
    """
    mean_petal_by_species = runtime.get_variable_value("mean_petal_by_species")
    
    if mean_petal_by_species is None:
        return ValidatorResult(
            success=False,
            message="Variable 'mean_petal_by_species' not found."
        )
    
    # Check if it's a pandas Series with correct structure
    if not isinstance(mean_petal_by_species, pd.Series):
        return ValidatorResult(
            success=False,
            message="Expected a pandas Series for mean_petal_by_species."
        )
    
    # Check values with tolerance
    tolerance = 0.01
    checks = [
        ('setosa', correct_answers["q1_setosa"]),
        ('versicolor', correct_answers["q1_versicolor"]),
        ('virginica', correct_answers["q1_virginica"])
    ]
    
    for species, expected in checks:
        if species not in mean_petal_by_species.index:
            return ValidatorResult(
                success=False,
                message=f"Missing species '{species}' in mean_petal_by_species."
            )
        
        actual = mean_petal_by_species[species]
        if abs(actual - expected) > tolerance:
            return ValidatorResult(
                success=False,
                message=f"Incorrect mean petal length for {species}. Expected ~{expected:.3f}, got {actual:.3f}."
            )
    
    return ValidatorResult(
        success=True,
        message=f"Correct. Mean petal lengths by species calculated successfully."
    )

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the extreme sepal length analysis.
    """
    max_sepal_sample = runtime.get_variable_value("max_sepal_sample")
    min_sepal_sample = runtime.get_variable_value("min_sepal_sample")
    feature_differences = runtime.get_variable_value("feature_differences")
    
    if any(var is None for var in [max_sepal_sample, min_sepal_sample, feature_differences]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for extreme samples analysis."
        )
    
    tolerance = 0.1
    
    # Check if correct samples were identified
    if hasattr(max_sepal_sample, 'name'):
        max_idx = max_sepal_sample.name
    else:
        max_idx = int(max_sepal_sample.index[0]) if hasattr(max_sepal_sample, 'index') else None
    
    if hasattr(min_sepal_sample, 'name'):
        min_idx = min_sepal_sample.name
    else:
        min_idx = int(min_sepal_sample.index[0]) if hasattr(min_sepal_sample, 'index') else None
    
    idx_correct = (max_idx == correct_answers["q2_max_sepal_idx"] and 
                   min_idx == correct_answers["q2_min_sepal_idx"])
    
    # Check feature differences
    sepal_diff = abs(feature_differences.get('sepal_length', 0))
    petal_diff = abs(feature_differences.get('petal_length', 0))
    
    sepal_correct = abs(sepal_diff - correct_answers["q2_sepal_length_diff"]) <= tolerance
    petal_correct = abs(petal_diff - correct_answers["q2_petal_length_diff"]) <= tolerance
    
    if idx_correct and sepal_correct and petal_correct:
        return ValidatorResult(
            success=True,
            message=f"Correct. Extreme samples identified and differences calculated."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect extreme sample analysis. Check sample identification and difference calculations."
        )

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the correlation analysis.
    """
    correlation_matrix = runtime.get_variable_value("correlation_matrix")
    highest_corr_pair = runtime.get_variable_value("highest_corr_pair")
    highest_corr_value = runtime.get_variable_value("highest_corr_value")
    
    if any(var is None for var in [correlation_matrix, highest_corr_pair, highest_corr_value]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for correlation analysis."
        )
    
    # Check correlation matrix structure
    if not isinstance(correlation_matrix, pd.DataFrame):
        return ValidatorResult(
            success=False,
            message="Expected a pandas DataFrame for correlation_matrix."
        )
    
    # Check highest correlation pair
    expected_pair = set(correct_answers["q3_highest_corr_pair"])
    actual_pair = set(highest_corr_pair) if isinstance(highest_corr_pair, (list, tuple)) else set()
    
    pair_correct = expected_pair == actual_pair
    value_correct = abs(highest_corr_value - correct_answers["q3_correlation_value"]) <= 0.01
    
    if pair_correct and value_correct:
        return ValidatorResult(
            success=True,
            message=f"Correct. Highest correlation: {highest_corr_value:.4f} between {highest_corr_pair}."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect correlation analysis. Expected pair {correct_answers['q3_highest_corr_pair']} with correlation {correct_answers['q3_correlation_value']:.4f}."
        )

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the comprehensive statistical analysis.
    """
    species_stats = runtime.get_variable_value("species_stats")
    cv_by_feature = runtime.get_variable_value("cv_by_feature")
    most_discriminative_feature = runtime.get_variable_value("most_discriminative_feature")
    
    if any(var is None for var in [species_stats, cv_by_feature, most_discriminative_feature]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for statistical analysis."
        )
    
    tolerance = 0.01
    
    try:
        # Check specific statistics
        setosa_petal_width_std = species_stats.loc['setosa', ('petal_width', 'std')]
        virginica_sepal_length_mean = species_stats.loc['virginica', ('sepal_length', 'mean')]
        
        std_correct = abs(setosa_petal_width_std - correct_answers["q4_setosa_petal_width_std"]) <= tolerance
        mean_correct = abs(virginica_sepal_length_mean - correct_answers["q4_virginica_sepal_length_mean"]) <= tolerance
        
        # Check most discriminative feature
        feature_correct = most_discriminative_feature == correct_answers["q4_most_variable_feature"]
        
        if std_correct and mean_correct and feature_correct:
            return ValidatorResult(
                success=True,
                message=f"Correct. Most discriminative feature: {most_discriminative_feature}."
            )
        else:
            return ValidatorResult(
                success=False,
                message="Incorrect statistical analysis. Check calculations for species statistics and CV."
            )
    except (KeyError, IndexError, AttributeError):
        return ValidatorResult(
            success=False,
            message="Error accessing statistics. Check the structure of species_stats."
        )

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the advanced shape analysis.
    """
    shape_features_df = runtime.get_variable_value("shape_features_df")
    f_ratios = runtime.get_variable_value("f_ratios")
    best_shape_feature = runtime.get_variable_value("best_shape_feature")
    best_feature_combination = runtime.get_variable_value("best_feature_combination")
    
    if any(var is None for var in [shape_features_df, f_ratios, best_shape_feature]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for shape analysis."
        )
    
    tolerance_ratio = 50.0  # Tolerance for F-ratio
    
    # Check if shape features were created correctly
    if 'petal_shape_ratio' not in shape_features_df.columns:
        return ValidatorResult(
            success=False,
            message="Missing 'petal_shape_ratio' in shape_features_df."
        )
    
    # Check best shape feature
    feature_correct = best_shape_feature == correct_answers["q5_best_shape_feature"]
    
    # Check F-ratio value (approximately)
    if isinstance(f_ratios, dict):
        best_f_ratio = f_ratios.get(best_shape_feature, 0)
    else:
        best_f_ratio = 0
    
    ratio_correct = abs(best_f_ratio - correct_answers["q5_best_f_ratio"]) <= tolerance_ratio
    
    # Check feature combination (if provided)
    combination_correct = True
    if best_feature_combination is not None:
        expected_combo = set(correct_answers["q5_best_combination"])
        actual_combo = set(best_feature_combination) if isinstance(best_feature_combination, (list, tuple)) else set()
        combination_correct = len(expected_combo.intersection(actual_combo)) >= 2  # At least 2 features match
    
    if feature_correct and (ratio_correct or combination_correct):
        return ValidatorResult(
            success=True,
            message=f"Correct. Best shape feature for species separation: {best_shape_feature}."
        )
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect shape analysis. Expected best feature: {correct_answers['q5_best_shape_feature']}."
        )

variables = [
    Variable(
        name="iris_df",
        value=iris_df,
        description="""A pandas DataFrame containing the Iris dataset.
        Columns and their dtypes:
        - id: int64 (Sample ID)
        - sepal_length: float64 (Sepal length in cm)
        - sepal_width: float64 (Sepal width in cm)
        - petal_length: float64 (Petal length in cm)
        - petal_width: float64 (Petal width in cm)
        - species: category (Species name: 'setosa', 'versicolor', 'virginica')
        
        Example: pd.DataFrame(data={
            'id': [1],
            'sepal_length': [5.1],
            'sepal_width': [3.5],
            'petal_length': [1.4],
            'petal_width': [0.2],
            'species': ['setosa']
        })
        Total samples: 150 (50 per species)
        You should use this variable to answer all questions.
        """
    ),
    Variable(
        name="mean_petal_by_species",
        value=pd.Series(dtype=float),
        description="Stores mean petal length for each species as a pandas Series."
    ),
    Variable(
        name="max_sepal_sample",
        value=pd.Series(dtype=float),
        description="Stores the sample (row) with maximum sepal length."
    ),
    Variable(
        name="min_sepal_sample",
        value=pd.Series(dtype=float),
        description="Stores the sample (row) with minimum sepal length."
    ),
    Variable(
        name="feature_differences",
        value={},
        description="Stores the differences in all features between max and min sepal length samples."
    ),
    Variable(
        name="correlation_matrix",
        value=pd.DataFrame(),
        description="Stores correlation matrix of numerical features in question 3."
    ),
    Variable(
        name="highest_corr_pair",
        value=[],
        description="Stores the names of the two features with highest positive correlation in question 3."
    ),
    Variable(
        name="highest_corr_value",
        value=0.0,
        description="Stores the highest positive correlation value found in the correlation matrix in question 3."
    ),
    Variable(
        name="species_stats",
        value=pd.DataFrame(),
        description="Stores comprehensive statistics for each feature by species."
    ),
    Variable(
        name="cv_by_feature",
        value={},
        description="Stores coefficient of variation for each feature across species."
    ),
    Variable(
        name="most_discriminative_feature",
        value="",
        description="Stores the name of the feature with highest coefficient of variation."
    ),
    Variable(
        name="shape_features_df",
        value=pd.DataFrame(),
        description="DataFrame containing original features plus new shape ratio features."
    ),
    Variable(
        name="f_ratios",
        value={},
        description="Stores F-ratios (between-species/within-species variance) for each feature."
    ),
    Variable(
        name="best_shape_feature",
        value="",
        description="Stores the name of the shape feature that best separates species."
    ),
    Variable(
        name="best_feature_combination",
        value=[],
        description="Stores the best combination of features for species separation."
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