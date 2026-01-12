"""
Data Injection Benchmark - Condition B (Runtime Injection)

Data is injected as a pandas DataFrame into Python runtime.
Agent accesses native Python objects - no serialization overhead.
Tests CaveAgent's ability to work with native data types.
"""

import os
from typing import List
import numpy as np
import pandas as pd
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# DATA PREPARATION
# ============================================================================

_data_dir = os.path.dirname(os.path.dirname(__file__))
_aapl_path = os.path.join(_data_dir, "data_analysis", "finance", "AAPL.csv")
_googl_path = os.path.join(_data_dir, "data_analysis", "finance", "GOOGL.csv")

# Load AAPL data - last 50 trading days
_aapl_full = pd.read_csv(_aapl_path, parse_dates=["Date"])
aapl_df = _aapl_full.sort_values("Date").tail(50).reset_index(drop=True)

# Load GOOGL data - last 50 trading days
_googl_full = pd.read_csv(_googl_path, parse_dates=["Date"])
googl_df = _googl_full.sort_values("Date").tail(50).reset_index(drop=True)

# Extract AAPL data for expected value computation
aapl_close = aapl_df["Close"].tolist()
aapl_volumes = aapl_df["Volume"].tolist()
aapl_opens = aapl_df["Open"].tolist()
aapl_highs = aapl_df["High"].tolist()
aapl_lows = aapl_df["Low"].tolist()

# Extract GOOGL data
googl_close = googl_df["Close"].tolist()
googl_volumes = googl_df["Volume"].tolist()
googl_opens = googl_df["Open"].tolist()
googl_highs = googl_df["High"].tolist()
googl_lows = googl_df["Low"].tolist()

# Pre-compute AAPL metrics
_aapl_daily_returns = np.diff(aapl_close) / np.array(aapl_close[:-1])
_aapl_daily_range = np.array(aapl_highs) - np.array(aapl_lows)
_aapl_intraday_change = np.array(aapl_close) - np.array(aapl_opens)

# Pre-compute GOOGL metrics
_googl_daily_returns = np.diff(googl_close) / np.array(googl_close[:-1])
_googl_daily_range = np.array(googl_highs) - np.array(googl_lows)

EXPECTED = {
    # Task 1: AAPL average daily range (High - Low)
    "avg_daily_range": float(np.mean(_aapl_daily_range)),
    # Task 2: AAPL total volume on up days (Close > Open)
    "up_day_volume": int(np.sum([v for c, o, v in zip(aapl_close, aapl_opens, aapl_volumes) if c > o])),
    # Task 3: AAPL correlation between daily range and volume
    "range_volume_corr": float(np.corrcoef(_aapl_daily_range, aapl_volumes)[0, 1]),
    # Task 4: AAPL average intraday return (Close - Open) / Open * 100
    "avg_intraday_return": float(np.mean(_aapl_intraday_change / np.array(aapl_opens) * 100)),
    # Task 5: AAPL volatility (std of daily returns) * sqrt(252) * 100
    "volatility": float(np.std(_aapl_daily_returns, ddof=1) * np.sqrt(252) * 100),
    # Task 6: AAPL max single-day percentage gain (Close - Open) / Open * 100
    "max_single_day_gain": float(np.max(_aapl_intraday_change / np.array(aapl_opens) * 100)),
    # Task 7: AAPL volume-weighted average price (sum(Close * Volume) / sum(Volume))
    "vwap": float(np.sum(np.array(aapl_close) * np.array(aapl_volumes)) / np.sum(aapl_volumes)),
    # Task 8: AAPL average range on down days (Close < Open)
    "down_day_avg_range": float(np.mean([h - l for c, o, h, l in zip(aapl_close, aapl_opens, aapl_highs, aapl_lows) if c < o])),
    # Task 9: Correlation between AAPL and GOOGL daily returns
    "aapl_googl_corr": float(np.corrcoef(_aapl_daily_returns, _googl_daily_returns)[0, 1]),
    # Task 10: Ratio of total AAPL volume to total GOOGL volume
    "volume_ratio": float(np.sum(aapl_volumes) / np.sum(googl_volumes)),
}


# ============================================================================
# VALIDATORS
# ============================================================================

def _validate_numeric(
    runtime: PythonRuntime,
    var_name: str,
    expected: float,
    tolerance: float = 0.001,
    is_integer: bool = False
) -> ValidatorResult:
    """Generic numeric validator with tolerance."""
    try:
        result = runtime.retrieve(var_name)

        if result is None:
            return ValidatorResult(False, f"{var_name} was not calculated")

        if is_integer:
            if int(result) != int(expected):
                return ValidatorResult(
                    False,
                    f"{var_name}: got {int(result)}, expected {int(expected)}"
                )
            return ValidatorResult(True, f"{var_name} = {int(result)} (exact match)")

        if expected != 0:
            rel_error = abs(result - expected) / abs(expected)
        else:
            rel_error = abs(result)

        if rel_error <= tolerance:
            return ValidatorResult(
                True,
                f"{var_name} = {result:.6f} (expected {expected:.6f}, error: {rel_error:.2e})"
            )
        else:
            return ValidatorResult(
                False,
                f"Precision loss: {var_name} = {result:.6f}, expected {expected:.6f}, "
                f"relative error: {rel_error:.2e} (tolerance: {tolerance})"
            )

    except Exception as e:
        return ValidatorResult(False, f"Error validating {var_name}: {str(e)}")


def validate_avg_daily_range(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "avg_daily_range", EXPECTED["avg_daily_range"])


def validate_up_day_volume(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "up_day_volume", EXPECTED["up_day_volume"], is_integer=True)


def validate_range_volume_corr(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "range_volume_corr", EXPECTED["range_volume_corr"], tolerance=0.01)


def validate_avg_intraday_return(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "avg_intraday_return", EXPECTED["avg_intraday_return"])


def validate_volatility(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "volatility", EXPECTED["volatility"], tolerance=0.01)


def validate_max_single_day_gain(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "max_single_day_gain", EXPECTED["max_single_day_gain"])


def validate_vwap(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "vwap", EXPECTED["vwap"])


def validate_down_day_avg_range(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "down_day_avg_range", EXPECTED["down_day_avg_range"])


def validate_aapl_googl_corr(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "aapl_googl_corr", EXPECTED["aapl_googl_corr"], tolerance=0.01)


def validate_volume_ratio(
    response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]
) -> ValidatorResult:
    return _validate_numeric(runtime, "volume_ratio", EXPECTED["volume_ratio"])


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

# DataFrames injected as runtime variables - agent has direct access
variables = [
    Variable("aapl_df", aapl_df,
        "AAPL stock DataFrame with 50 rows. Columns: Date, Open, High, Low, Close, Volume."),
    Variable("googl_df", googl_df,
        "GOOGL stock DataFrame with 50 rows. Columns: Date, Open, High, Low, Close, Volume."),

    # Output variables
    Variable("avg_daily_range", None, "Store AAPL average daily range (High-Low) here (float)."),
    Variable("up_day_volume", None, "Store AAPL total volume on up days here (int)."),
    Variable("range_volume_corr", None, "Store AAPL range-volume correlation here (float)."),
    Variable("avg_intraday_return", None, "Store AAPL average intraday return % here (float)."),
    Variable("volatility", None, "Store AAPL annualized volatility % here (float)."),
    Variable("max_single_day_gain", None, "Store AAPL max single-day gain % here (float)."),
    Variable("vwap", None, "Store AAPL volume-weighted average price here (float)."),
    Variable("down_day_avg_range", None, "Store AAPL average range on down days here (float)."),
    Variable("aapl_googl_corr", None, "Store AAPL-GOOGL return correlation here (float)."),
    Variable("volume_ratio", None, "Store AAPL/GOOGL volume ratio here (float)."),
]

validators = {
    "validate_avg_daily_range": validate_avg_daily_range,
    "validate_up_day_volume": validate_up_day_volume,
    "validate_range_volume_corr": validate_range_volume_corr,
    "validate_avg_intraday_return": validate_avg_intraday_return,
    "validate_volatility": validate_volatility,
    "validate_max_single_day_gain": validate_max_single_day_gain,
    "validate_vwap": validate_vwap,
    "validate_down_day_avg_range": validate_down_day_avg_range,
    "validate_aapl_googl_corr": validate_aapl_googl_corr,
    "validate_volume_ratio": validate_volume_ratio,
}

hooks = {}
