"""
AAPL vs GOOGL Comparative Analysis - Multi-Turn Benchmark

Tests:
- Handling large DataFrames (10K+ rows input)
- Computing scalar values (floats, strings)
- DataFrame merging
- Large output DataFrame (~5000 rows)
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

_data_dir = os.path.dirname(__file__)
aapl_df = pd.read_csv(os.path.join(_data_dir, "AAPL.csv"), parse_dates=["Date"])
googl_df = pd.read_csv(os.path.join(_data_dir, "GOOGL.csv"), parse_dates=["Date"])


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_total_returns(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Turn 1: Calculate total returns (floats) and better performer (string)."""
    try:
        aapl_return = runtime.get_variable("aapl_total_return")
        googl_return = runtime.get_variable("googl_total_return")
        better = runtime.get_variable("better_performer")

        errors = []

        # Expected values
        expected_aapl = (aapl_df["Close"].iloc[-1] - aapl_df["Close"].iloc[0]) / aapl_df["Close"].iloc[0] * 100
        expected_googl = (googl_df["Close"].iloc[-1] - googl_df["Close"].iloc[0]) / googl_df["Close"].iloc[0] * 100
        expected_better = "AAPL" if expected_aapl > expected_googl else "GOOGL"

        if aapl_return is None:
            errors.append("aapl_total_return not calculated")
        elif abs(aapl_return - expected_aapl) / expected_aapl > 0.01:
            errors.append(f"aapl_total_return: {aapl_return:.2f}%, expected {expected_aapl:.2f}%")

        if googl_return is None:
            errors.append("googl_total_return not calculated")
        elif abs(googl_return - expected_googl) / expected_googl > 0.01:
            errors.append(f"googl_total_return: {googl_return:.2f}%, expected {expected_googl:.2f}%")

        if better is None:
            errors.append("better_performer not set")
        elif better.upper() != expected_better:
            errors.append(f"better_performer: '{better}', expected '{expected_better}'")

        if errors:
            return ValidatorResult(False, "; ".join(errors))

        return ValidatorResult(True, f"AAPL: {aapl_return:.2f}%, GOOGL: {googl_return:.2f}%, Winner: {better}")

    except Exception as e:
        return ValidatorResult(False, str(e))


def validate_merged_data(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Turn 2: Merge DataFrames on overlapping dates."""
    try:
        merged = runtime.get_variable("merged_df")

        if merged is None:
            return ValidatorResult(False, "merged_df not created")

        if not isinstance(merged, pd.DataFrame):
            return ValidatorResult(False, f"merged_df should be DataFrame, got {type(merged).__name__}")

        errors = []

        # Check columns
        required = ["Date", "aapl_close", "googl_close"]
        missing = [c for c in required if c not in merged.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Check size (~5000 overlapping dates)
        expected_size = len(pd.merge(aapl_df[["Date"]], googl_df[["Date"]], on="Date"))
        if abs(len(merged) - expected_size) > 50:
            errors.append(f"Expected ~{expected_size} rows, got {len(merged)}")

        if errors:
            return ValidatorResult(False, "; ".join(errors))

        return ValidatorResult(True, f"Merged DataFrame: {len(merged)} rows, columns: {list(merged.columns)}")

    except Exception as e:
        return ValidatorResult(False, str(e))


def validate_daily_returns(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Turn 3: Add daily returns and compute correlation."""
    try:
        merged = runtime.get_variable("merged_df")
        corr = runtime.get_variable("correlation")

        errors = []

        if merged is None:
            errors.append("merged_df not found")
        elif not isinstance(merged, pd.DataFrame):
            errors.append(f"merged_df should be DataFrame, got {type(merged).__name__}")
        else:
            if "aapl_return" not in merged.columns:
                errors.append("aapl_return column not added")
            if "googl_return" not in merged.columns:
                errors.append("googl_return column not added")

        if corr is None:
            errors.append("correlation not calculated")
        elif not isinstance(corr, (int, float)):
            errors.append(f"correlation should be numeric, got {type(corr).__name__}")
        elif corr < -1 or corr > 1:
            errors.append(f"correlation should be between -1 and 1, got {corr}")

        if errors:
            return ValidatorResult(False, "; ".join(errors))

        return ValidatorResult(True, f"Daily returns added. Correlation: {corr:.4f}")

    except Exception as e:
        return ValidatorResult(False, str(e))


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("aapl_df", aapl_df,
        "AAPL stock data (10K rows, 1984-2025). Columns: Date, Open, High, Low, Close, Volume."),
    Variable("googl_df", googl_df,
        "GOOGL stock data (5K rows, 2004-2025). Columns: Date, Open, High, Low, Close, Volume."),
    Variable("aapl_total_return", None,
        "AAPL total return percentage (float)."),
    Variable("googl_total_return", None,
        "GOOGL total return percentage (float)."),
    Variable("better_performer", None,
        "Ticker of better performer (string: 'AAPL' or 'GOOGL')."),
    Variable("merged_df", None,
        "Merged DataFrame with columns: Date, aapl_close, googl_close."),
    Variable("correlation", None,
        "Correlation between daily returns (float)."),
]

validators = {
    "validate_total_returns": validate_total_returns,
    "validate_merged_data": validate_merged_data,
    "validate_daily_returns": validate_daily_returns,
}
