"""
AAPL Stock Analysis - Multi-Turn Benchmark

Tests:
- Basic statistical calculations on stock data
- Filtering based on previous results
- Comparative analysis using prior computations
- Correlation analysis building on full context
"""

import os
from typing import List
import pandas as pd
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# PRE-LOADED DATA
# ============================================================================

_data_path = os.path.join(os.path.dirname(__file__), "AAPL_20240101_20241231.csv")
df = pd.read_csv(_data_path, parse_dates=["Date"])


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_avg_close(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 1: Calculate average closing price.
    Expected: avg_close ≈ 207.94 (within tolerance)
    """
    try:
        avg_close = runtime.get_variable("avg_close")
        expected = df["Close"].mean()

        if avg_close is None:
            return ValidatorResult(False, "avg_close was not calculated")

        if not isinstance(avg_close, (int, float)):
            return ValidatorResult(False, f"avg_close should be numeric, got {type(avg_close).__name__}")

        tolerance = 0.01
        if abs(avg_close - expected) / expected > tolerance:
            return ValidatorResult(False, f"avg_close is {avg_close:.2f}, expected {expected:.2f}")

        return ValidatorResult(True, f"Correct! Average closing price: ${avg_close:.2f}")

    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_days_above_avg(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 2: Count days where close > avg_close from Turn 1.
    Must use the avg_close variable from previous turn.
    """
    try:
        avg_close = runtime.get_variable("avg_close")
        days_above = runtime.get_variable("days_above_avg")

        if avg_close is None:
            return ValidatorResult(False, "avg_close from previous turn not found")

        if days_above is None:
            return ValidatorResult(False, "days_above_avg was not calculated")

        expected = int((df["Close"] > avg_close).sum())

        # if not isinstance(days_above, (int, float)):
        #     return ValidatorResult(False, f"days_above_avg should be numeric, got {type(days_above).__name__}")

        if int(days_above) != expected:
            return ValidatorResult(False, f"days_above_avg is {int(days_above)}, expected {expected}")

        return ValidatorResult(True, f"Correct! {expected} days closed above the average of ${avg_close:.2f}")

    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_volume_comparison(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 3: Compare volume on above-average days vs overall.
    Must use avg_close from Turn 1 to filter days.
    """
    try:
        avg_close = runtime.get_variable("avg_close")
        above_avg_vol = runtime.get_variable("above_avg_volume")
        overall_vol = runtime.get_variable("overall_avg_volume")
        ratio = runtime.get_variable("volume_ratio")

        if avg_close is None:
            return ValidatorResult(False, "avg_close from Turn 1 not found")

        errors = []

        # Calculate expected values
        above_avg_days = df[df["Close"] > avg_close]
        expected_above_vol = above_avg_days["Volume"].mean()
        expected_overall_vol = df["Volume"].mean()
        expected_ratio = expected_above_vol / expected_overall_vol

        tolerance = 0.01

        if above_avg_vol is None:
            errors.append("above_avg_volume was not calculated")
        elif abs(above_avg_vol - expected_above_vol) / expected_above_vol > tolerance:
            errors.append(f"above_avg_volume is {above_avg_vol:.0f}, expected {expected_above_vol:.0f}")

        if overall_vol is None:
            errors.append("overall_avg_volume was not calculated")
        elif abs(overall_vol - expected_overall_vol) / expected_overall_vol > tolerance:
            errors.append(f"overall_avg_volume is {overall_vol:.0f}, expected {expected_overall_vol:.0f}")

        if ratio is None:
            errors.append("volume_ratio was not calculated")
        elif abs(ratio - expected_ratio) / expected_ratio > tolerance:
            errors.append(f"volume_ratio is {ratio:.3f}, expected {expected_ratio:.3f}")

        if errors:
            return ValidatorResult(False, "; ".join(errors))

        trend = "higher" if ratio > 1 else "lower"
        return ValidatorResult(True,
            f"Correct! Above-average days had {trend} volume (ratio: {ratio:.3f})")

    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_correlation(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 4: Calculate correlation between price and volume.
    """
    try:
        corr = runtime.get_variable("price_volume_corr")

        if corr is None:
            return ValidatorResult(False, "price_volume_corr was not calculated")

        if not isinstance(corr, (int, float)):
            return ValidatorResult(False, f"price_volume_corr should be numeric, got {type(corr).__name__}")

        expected = df["Close"].corr(df["Volume"])

        tolerance = 0.01
        if abs(corr - expected) > tolerance:
            return ValidatorResult(False, f"price_volume_corr is {corr:.4f}, expected {expected:.4f}")

        # Interpret the correlation
        if corr > 0.3:
            interpretation = "positive correlation - higher prices with higher volume"
        elif corr < -0.3:
            interpretation = "negative correlation - higher prices with lower volume"
        else:
            interpretation = "weak correlation - no strong relationship"

        return ValidatorResult(True, f"Correct! Correlation: {corr:.4f} ({interpretation})")

    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable(
        "df", df,
        "AAPL stock DataFrame with columns: Date, Open, High, Low, Close, Volume. "
        "Contains daily trading data for 2024 (252 trading days)."
    ),
    Variable(
        "avg_close", None,
        "Store the average closing price here (float)."
    ),
    Variable(
        "days_above_avg", None,
        "Store the count of days closing above avg_close here (int)."
    ),
    Variable(
        "above_avg_volume", None,
        "Store the average volume for above-average price days here (float)."
    ),
    Variable(
        "overall_avg_volume", None,
        "Store the overall average daily volume here (float)."
    ),
    Variable(
        "volume_ratio", None,
        "Store the ratio of above_avg_volume to overall_avg_volume here (float)."
    ),
    Variable(
        "price_volume_corr", None,
        "Store the correlation coefficient between Close and Volume here (float)."
    ),
]

validators = {
    "validate_avg_close": validate_avg_close,
    "validate_days_above_avg": validate_days_above_avg,
    "validate_volume_comparison": validate_volume_comparison,
    "validate_correlation": validate_correlation,
}
