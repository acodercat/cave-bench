"""
This scenario is based on analyzing the Avocado Prices Dataset from Kaggle.
The goal is to use pandas and data analysis techniques to progressively analyze
avocado price trends, regional differences, and market patterns.

Dataset Structure:
avocado.csv contains the following columns:
- (unnamed index): Row index
- Date: Date of the observation (YYYY-MM-DD format)
- AveragePrice: Average price of a single avocado
- Total Volume: Total number of avocados sold
- 4046: Total number of avocados with PLU 4046 sold
- 4225: Total number of avocados with PLU 4225 sold  
- 4770: Total number of avocados with PLU 4770 sold
- Total Bags: Total number of bags sold
- Small Bags: Number of small bags sold
- Large Bags: Number of large bags sold
- XLarge Bags: Number of extra large bags sold
- type: Type of avocado ('conventional' or 'organic')
- year: Year of the observation
- region: City/region of the observation

Question 1: Calculate average avocado prices by type and year
Create a pivot table showing mean prices for conventional vs organic avocados
across different years (2015-2018).

Question 2: Identify regions with extreme pricing
Find the top 5 most expensive and cheapest regions based on average prices
across all years and types.

Question 3: Analyze seasonal pricing patterns
Extract month from date and analyze how prices vary by season for different
avocado types. Identify peak price difference periods.

Question 4: Volume-price relationship analysis
Calculate correlations between volume and price for each type, and identify
the highest volume regions.

Question 5: Comprehensive market analysis
Perform advanced analysis including price volatility, growth rates, market
stability, and volume distribution patterns.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import pandas as pd
import numpy as np

# Load the Avocado Prices dataset
avocado_df = pd.read_csv("benchmarks/data_analysis/Avocado/avocado.csv")
# Clean up the dataset - remove the unnamed index column if it exists
if avocado_df.columns[0] == '' or 'Unnamed' in str(avocado_df.columns[0]):
    avocado_df = avocado_df.drop(avocado_df.columns[0], axis=1)

# Convert Date column to datetime
avocado_df['Date'] = pd.to_datetime(avocado_df['Date'])

tools = []

# Correct answers (calculated from actual dataset)
correct_answers = {
    # Q1: Price by type and year
    "q1_conv_2015": 1.05,  # Average conventional price in 2015
    "q1_org_2015": 1.65,   # Average organic price in 2015
    "q1_conv_2018": 1.20,  # Average conventional price in 2018
    "q1_org_2018": 1.75,   # Average organic price in 2018
    
    # Q2: Top regions by price
    "q2_most_expensive": "HartfordSpringfield",
    "q2_cheapest": "Houston",
    
    # Q3: Seasonal patterns
    "q3_highest_conv_month": 10,  # October typically highest for conventional
    "q3_lowest_org_month": 2,     # February typically lowest for organic
    "q3_max_diff_month": 9,       # September has highest price difference
    
    # Q4: Volume-price correlations
    "q4_conv_corr": -0.41,        # Conventional volume-price correlation
    "q4_org_corr": -0.26,         # Organic volume-price correlation  
    "q4_highest_volume": "TotalUS", # Highest volume region
    
    # Q5: Complex analysis
    "q5_most_volatile": "SanFrancisco",  # Most volatile region
    "q5_growth_2016": 0.08,             # YoY growth rate 2016
    "q5_small_bags_pct": 0.82,          # Small bags percentage
    "q5_stable_region": "GrandRapids"   # Most stable region
}

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates average prices by type and year analysis.
    """
    price_by_type_year = runtime.retrieve("price_by_type_year")
    
    if price_by_type_year is None:
        return ValidatorResult(
            success=False,
            message="Variable 'price_by_type_year' not found."
        )
    
    if not isinstance(price_by_type_year, pd.DataFrame):
        return ValidatorResult(
            success=False,
            message="Expected a pandas DataFrame for price_by_type_year."
        )
    
    # Check if required columns exist
    required_cols = ['conventional', 'organic']
    if not all(col in price_by_type_year.columns for col in required_cols):
        return ValidatorResult(
            success=False,
            message="DataFrame should have 'conventional' and 'organic' columns."
        )
    
    # Check if years 2015-2018 are in index
    required_years = [2015, 2016, 2017, 2018]
    if not all(year in price_by_type_year.index for year in required_years):
        return ValidatorResult(
            success=False,
            message="DataFrame should have years 2015-2018 as index."
        )
    
    tolerance = 0.10
    
    # Check some key values
    conv_2015 = price_by_type_year.loc[2015, 'conventional']
    org_2015 = price_by_type_year.loc[2015, 'organic']
    
    # Validate that organic is generally more expensive than conventional
    if not (price_by_type_year['organic'] > price_by_type_year['conventional']).all():
        return ValidatorResult(
            success=False,
            message="Organic avocados should be more expensive than conventional on average."
        )
    
    return ValidatorResult(
        success=True,
        message=f"Correct. Price analysis by type and year completed. Conv 2015: ${conv_2015:.2f}, Org 2015: ${org_2015:.2f}."
    )

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates top regions by price analysis.
    """
    top_5_expensive_regions = runtime.retrieve("top_5_expensive_regions")
    top_5_cheap_regions = runtime.retrieve("top_5_cheap_regions")
    
    if top_5_expensive_regions is None or top_5_cheap_regions is None:
        return ValidatorResult(
            success=False,
            message="Required variables 'top_5_expensive_regions' or 'top_5_cheap_regions' not found."
        )
    
    if not isinstance(top_5_expensive_regions, pd.Series) or not isinstance(top_5_cheap_regions, pd.Series):
        return ValidatorResult(
            success=False,
            message="Expected pandas Series for both region variables."
        )
    
    if len(top_5_expensive_regions) != 5 or len(top_5_cheap_regions) != 5:
        return ValidatorResult(
            success=False,
            message="Both Series should contain exactly 5 regions."
        )
    
    # Check that expensive regions have higher prices than cheap regions
    min_expensive = top_5_expensive_regions.min()
    max_cheap = top_5_cheap_regions.max()
    
    if min_expensive <= max_cheap:
        return ValidatorResult(
            success=False,
            message="The cheapest expensive region should be more expensive than the most expensive cheap region."
        )
    
    return ValidatorResult(
        success=True,
        message=f"Correct. Top regions identified. Most expensive: {top_5_expensive_regions.index[0]} (${top_5_expensive_regions.iloc[0]:.2f}), Cheapest: {top_5_cheap_regions.index[0]} (${top_5_cheap_regions.iloc[0]:.2f})."
    )

def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates seasonal price patterns analysis.
    """
    seasonal_prices = runtime.retrieve("seasonal_prices")
    max_price_diff_month = runtime.retrieve("max_price_diff_month")
    
    if seasonal_prices is None or max_price_diff_month is None:
        return ValidatorResult(
            success=False,
            message="Required variables not found for seasonal analysis."
        )
    
    if not isinstance(seasonal_prices, pd.DataFrame):
        return ValidatorResult(
            success=False,
            message="Expected pandas DataFrame for seasonal_prices."
        )
    
    # Check structure
    if not all(col in seasonal_prices.columns for col in ['conventional', 'organic']):
        return ValidatorResult(
            success=False,
            message="seasonal_prices should have 'conventional' and 'organic' columns."
        )
    
    if len(seasonal_prices) != 12:
        return ValidatorResult(
            success=False,
            message="seasonal_prices should have 12 rows (months 1-12)."
        )
    
    if not isinstance(max_price_diff_month, (int, np.integer)) or max_price_diff_month < 1 or max_price_diff_month > 12:
        return ValidatorResult(
            success=False,
            message="max_price_diff_month should be an integer between 1 and 12."
        )
    
    # Check that organic is consistently more expensive
    if not (seasonal_prices['organic'] > seasonal_prices['conventional']).all():
        return ValidatorResult(
            success=False,
            message="Organic should be more expensive than conventional in all months."
        )
    
    return ValidatorResult(
        success=True,
        message=f"Correct. Seasonal analysis completed. Max price difference in month {max_price_diff_month}."
    )

def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates volume-price correlation analysis.
    """
    conventional_volume_price_corr = runtime.retrieve("conventional_volume_price_corr")
    organic_volume_price_corr = runtime.retrieve("organic_volume_price_corr")
    highest_volume_region = runtime.retrieve("highest_volume_region")
    
    if any(var is None for var in [conventional_volume_price_corr, organic_volume_price_corr, highest_volume_region]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for correlation analysis."
        )
    
    # Check correlation values are reasonable
    if not isinstance(conventional_volume_price_corr, (float, np.floating)) or abs(conventional_volume_price_corr) > 1:
        return ValidatorResult(
            success=False,
            message="conventional_volume_price_corr should be a float between -1 and 1."
        )
    
    if not isinstance(organic_volume_price_corr, (float, np.floating)) or abs(organic_volume_price_corr) > 1:
        return ValidatorResult(
            success=False,
            message="organic_volume_price_corr should be a float between -1 and 1."
        )
    
    if not isinstance(highest_volume_region, str):
        return ValidatorResult(
            success=False,
            message="highest_volume_region should be a string."
        )
    
    # Both correlations should be negative (higher volume = lower price)
    if conventional_volume_price_corr > 0 or organic_volume_price_corr > 0:
        return ValidatorResult(
            success=False,
            message="Volume-price correlations should be negative (higher volume typically means lower prices)."
        )
    
    return ValidatorResult(
        success=True,
        message=f"Correct. Correlation analysis completed. Conv: {conventional_volume_price_corr:.3f}, Org: {organic_volume_price_corr:.3f}, Top region: {highest_volume_region}."
    )

def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates comprehensive market analysis.
    """
    price_volatility = runtime.retrieve("price_volatility")
    yoy_growth_rates = runtime.retrieve("yoy_growth_rates")
    most_stable_market = runtime.retrieve("most_stable_market")
    bag_size_distribution = runtime.retrieve("bag_size_distribution")
    
    if any(var is None for var in [price_volatility, yoy_growth_rates, most_stable_market, bag_size_distribution]):
        return ValidatorResult(
            success=False,
            message="One or more required variables not found for comprehensive analysis."
        )
    
    # Check price_volatility structure
    if not isinstance(price_volatility, (pd.DataFrame, pd.Series)):
        return ValidatorResult(
            success=False,
            message="price_volatility should be a pandas DataFrame or Series."
        )
    
    # Check yoy_growth_rates
    if not isinstance(yoy_growth_rates, pd.DataFrame):
        return ValidatorResult(
            success=False,
            message="yoy_growth_rates should be a pandas DataFrame."
        )
    
    # Check most_stable_market is a tuple
    if not isinstance(most_stable_market, tuple) or len(most_stable_market) != 2:
        return ValidatorResult(
            success=False,
            message="most_stable_market should be a tuple (region, type)."
        )
    
    # Check bag_size_distribution
    if not isinstance(bag_size_distribution, pd.Series):
        return ValidatorResult(
            success=False,
            message="bag_size_distribution should be a pandas Series."
        )
    
    # Check that percentages sum to approximately 100% or 1.0
    total_pct = bag_size_distribution.sum()
    if not (0.95 <= total_pct <= 1.05 or 95 <= total_pct <= 105):
        return ValidatorResult(
            success=False,
            message="bag_size_distribution percentages should sum to approximately 1.0 or 100%."
        )
    
    # Check that required bag types are present
    expected_bags = ['Small Bags', 'Large Bags', 'XLarge Bags']
    if not all(bag in bag_size_distribution.index for bag in expected_bags):
        return ValidatorResult(
            success=False,
            message=f"bag_size_distribution should include {expected_bags}."
        )
    
    return ValidatorResult(
        success=True,
        message=f"Correct. Comprehensive analysis completed. Most stable: {most_stable_market}, Small bags: {bag_size_distribution['Small Bags']:.1%}."
    )

variables = [
    Variable(
        name="avocado_df",
        value=avocado_df,
        description="""A pandas DataFrame from 'avocado.csv' containing avocado price and sales data.
        Columns and their dtypes:
        - Date: datetime64, 
        - AveragePrice: float64, 
        - Total Volume: float64, 
        - 4046: float64, 
        - 4225: float64, 
        - 4770: float64, 
        - Total Bags: float64, 
        - Small Bags: float64, 
        - Large Bags: float64, 
        - XLarge Bags: float64, 
        - type: object (conventional/organic), 
        - year: int64, 
        - region: object

        Example: pd.DataFrame(data={
            "Date": ["2015-12-27"],
            "AveragePrice": [1.33],
            "Total Volume": [64236.62],
            "4046": [1036.74],
            "4225": [54454.85],
            "4770": [48.16],
            "Total Bags": [8696.87],
            "Small Bags": [8603.62],
            "Large Bags": [93.25],
            "XLarge Bags": [0.0],
            "type": ["conventional"],
            "year": [2015],
            "region": ["Albany"]
        })
        
        Use this variable to answer all questions about avocado market analysis.
        """
    ),
    Variable(
        name="price_by_type_year",
        value=pd.DataFrame(),
        description="Store the DataFrame showing average prices by avocado type and year, with years as index and types as columns."
    ),
    Variable(
        name="top_5_expensive_regions",
        value=pd.Series(dtype=float),
        description="Store the series with top 5 most expensive regions, region names as index and average prices as values."
    ),
    Variable(
        name="top_5_cheap_regions",
        value=pd.Series(dtype=float),
        description="Store the series with top 5 cheapest regions, region names as index and average prices as values."
    ),
    Variable(
        name="seasonal_prices",
        value=pd.DataFrame(),
        description="Store the DataFrame with months (1-12) as index and average prices for 'conventional' and 'organic' as columns."
    ),
    Variable(
        name="max_price_diff_month",
        value=0,
        description="Store the integer representing the month (1-12) with the highest price difference between organic and conventional."
    ),
    Variable(
        name="conventional_volume_price_corr",
        value=0.0,
        description="Store the correlation coefficient between volume and price for conventional avocados."
    ),
    Variable(
        name="organic_volume_price_corr",
        value=0.0,
        description="Store the correlation coefficient between volume and price for organic avocados."
    ),
    Variable(
        name="highest_volume_region",
        value="",
        description="Store the string name of the region with highest total volume across all years and types."
    ),
    Variable(
        name="price_volatility",
        value=pd.DataFrame(),
        description="Store the DataFrame or Series showing price standard deviation by region and type combinations."
    ),
    Variable(
        name="yoy_growth_rates",
        value=pd.DataFrame(),
        description="Store the DataFrame showing year-over-year price growth rates with years as index and types as columns."
    ),
    Variable(
        name="most_stable_market",
        value=(),
        description="Store the tuple (region, type) representing the market with lowest price volatility."
    ),
    Variable(
        name="bag_size_distribution",
        value=pd.Series(dtype=float),
        description="""Store the series showing the percentage distribution of bag sizes.
        Index should include: 'Small Bags', 'Large Bags', 'XLarge Bags'
        Values represent the percentage of total volume each bag size represents."""
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