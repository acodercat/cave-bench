"""Comprehensive Stock Data Benchmark - JSON Function Calling Version

30 queries across 3 categories:
- Data Query (10): Use provided query tools (return JSON)
- Data Analysis (10): NO tools provided - FC cannot write code!
- Data Visualization (10): NO tools provided - FC cannot write code!

FC Limitation: Can only call predefined functions. Cannot write arbitrary code.
This demonstrates FC's inflexibility for open-ended analysis/visualization tasks.
"""

import json
from typing import List, Dict, Any
from core.validation import ValidatorResult
from utils import extract_json_from_response

# Import data and base tools from main module to ensure consistency
from .stock_benchmark import (
    AAPL_DF, GOOGL_DF, EXPECTED,
    query_by_date_range as _query_by_date_range,
    query_by_volume_threshold as _query_by_volume_threshold,
    query_top_volume_days as _query_top_volume_days,
    query_price_change_days as _query_price_change_days,
    query_last_n_days as _query_last_n_days,
    query_weekly_ohlc as _query_weekly_ohlc,
    query_monthly_first_last as _query_monthly_first_last,
    query_52week_high_days as _query_52week_high_days,
    query_cross_stock_condition as _query_cross_stock_condition,
)


# ============================================================================
# JSON WRAPPER FUNCTIONS - Wrap base tools to return JSON for FC
# ============================================================================

def _df_to_json(df) -> List[Dict[str, Any]]:
    """Convert DataFrame to JSON-serializable list of records."""
    import pandas as pd
    result = df.copy()
    # Convert datetime columns to string format
    for col in ["Date", "First_Date", "Last_Date"]:
        if col in result.columns and pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = result[col].dt.strftime("%Y-%m-%d")
    return result.to_dict(orient="records")


def query_by_date_range(symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Query stock data for a date range.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'

    Returns:
        List of records with keys: Date, Open, High, Low, Close, Volume
        Example: [{"Date": "2023-01-03", "Open": 130.28, "High": 130.90, "Low": 124.17, "Close": 125.07, "Volume": 112117500}, ...]
    """
    return _df_to_json(_query_by_date_range(symbol, start_date, end_date))


def query_by_volume_threshold(symbol: str, min_volume: float) -> List[Dict[str, Any]]:
    """Find days where volume exceeded threshold.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        min_volume: Minimum volume threshold (e.g., 50000000 for 50 million)

    Returns:
        List of records with keys: Date, Open, High, Low, Close, Volume
        Contains only records where Volume > min_volume
    """
    return _df_to_json(_query_by_volume_threshold(symbol, min_volume))


def query_top_volume_days(symbol: str, n: int) -> List[Dict[str, Any]]:
    """Get top N highest volume trading days.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        n: Number of top days to return

    Returns:
        List of records with keys: Date, Open, High, Low, Close, Volume
        Sorted by Volume descending, limited to n records
    """
    return _df_to_json(_query_top_volume_days(symbol, n))


def query_price_change_days(symbol: str, min_pct_change: float) -> List[Dict[str, Any]]:
    """Find days where close > open by percentage.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        min_pct_change: Minimum percentage change (e.g., 5 for 5%)

    Returns:
        List of records with keys: Date, Open, High, Low, Close, Volume
        Contains only records where (Close - Open) / Open * 100 > min_pct_change
    """
    return _df_to_json(_query_price_change_days(symbol, min_pct_change))


def query_last_n_days(symbol: str, n: int) -> List[Dict[str, Any]]:
    """Get last N trading days.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        n: Number of most recent trading days to return

    Returns:
        List of records with keys: Date, Open, High, Low, Close, Volume
        Contains the last n records sorted by Date ascending
    """
    return _df_to_json(_query_last_n_days(symbol, n))


def query_weekly_ohlc(symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Get weekly OHLC data (resampled from daily data).

    Args:
        symbol: 'AAPL' or 'GOOGL'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'

    Returns:
        List of records with keys: Date, Open, High, Low, Close, Volume
        Each record represents one week: Open=first day, High=max, Low=min, Close=last day, Volume=sum
    """
    return _df_to_json(_query_weekly_ohlc(symbol, start_date, end_date))


def query_monthly_first_last(symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Get monthly first and last trading day prices.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'

    Returns:
        List of records with keys: Month, First_Date, Last_Date, First_Open, Last_Close
        Each record represents one month with first/last trading day data
    """
    return _df_to_json(_query_monthly_first_last(symbol, start_date, end_date))


def query_52week_high_days(symbol: str) -> List[Dict[str, Any]]:
    """Find days where stock hit 52-week high.

    Args:
        symbol: 'AAPL' or 'GOOGL'

    Returns:
        List of records with keys: Date, High, Close, Volume
        Contains only days where the High price equals or exceeds the rolling 252-day maximum
    """
    return _df_to_json(_query_52week_high_days(symbol))


def query_cross_stock_condition(condition: str) -> List[Dict[str, Any]]:
    """Cross-reference days based on condition comparing AAPL and GOOGL.

    Args:
        condition: 'aapl_up_googl_down' (AAPL up, GOOGL down) or 'both_down_3pct' (both down >3%)

    Returns:
        List of records with keys: Date, Close_AAPL, AAPL_Return, Close_GOOGL, GOOGL_Return
        Contains only days matching the specified condition
    """
    return _df_to_json(_query_cross_stock_condition(condition))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _json_data_hash(data: list) -> str:
    """Compute hash of JSON data list for exact matching."""
    import hashlib
    import pandas as pd
    import numpy as np
    # Convert to DataFrame for consistent hashing
    df = pd.DataFrame(data)
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)
        df["Date"] = df["Date"].astype(str)
    # Round numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].round(4)
    content = df.to_csv(index=False)
    return hashlib.md5(content.encode()).hexdigest()


def _check_data_hash(response: str, expected_hash: str, expected_count: int, query_name: str) -> ValidatorResult:
    """Check if response data matches expected hash (exact match)."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, f"{query_name}: Could not extract JSON from response")
    data = result.get("data", result.get("queried_data", []))
    if not isinstance(data, list):
        return ValidatorResult(False, f"{query_name}: 'data' must be a list")
    if len(data) != expected_count:
        return ValidatorResult(False, f"{query_name}: Expected {expected_count} rows, got {len(data)}")
    actual_hash = _json_data_hash(data)
    if actual_hash == expected_hash:
        return ValidatorResult(True, f"{query_name}: Data hash matches ({len(data)} rows)")
    return ValidatorResult(False, f"{query_name}: Data hash mismatch (got {actual_hash[:8]}..., expected {expected_hash[:8]}...)")


def _check_chart_type(response: str, expected_type: str, query_name: str, min_series: int = 1) -> ValidatorResult:
    """Check if chart config has expected type and series count."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, f"{query_name}: Could not extract JSON from response")
    config = result.get("chart_config", result)
    if not isinstance(config, dict):
        return ValidatorResult(False, f"{query_name}: chart_config must be a dict")
    if "series" not in config:
        return ValidatorResult(False, f"{query_name}: Missing 'series' in chart config")
    series = config["series"]
    if not series:
        return ValidatorResult(False, f"{query_name}: Empty series")
    chart_type = series[0].get("type", "")
    if chart_type != expected_type:
        return ValidatorResult(False, f"{query_name}: Expected {expected_type}, got {chart_type}")
    if len(series) < min_series:
        return ValidatorResult(False, f"{query_name}: Expected >= {min_series} series, got {len(series)}")
    return ValidatorResult(True, f"{query_name}: Valid {chart_type} chart with {len(series)} series")


# ============================================================================
# DATA QUERY VALIDATORS (Q1-Q10)
# ============================================================================

def validate_q1(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q1: AAPL January 2023 data - exact hash match."""
    exp = EXPECTED["q1"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q1 (AAPL Jan 2023)")


def validate_q2(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q2: GOOGL volume > 100M days - exact hash match."""
    exp = EXPECTED["q2"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q2 (GOOGL Vol>100M)")


def validate_q3(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q3: Both AAPL and GOOGL down > 3% - exact hash match."""
    exp = EXPECTED["q3"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q3 (Both down >3%)")


def validate_q4(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q4: Top 5 AAPL volume days - exact hash match."""
    exp = EXPECTED["q4"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q4 (Top 5 volume)")


def validate_q5(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q5: GOOGL weekly OHLC Q4 2023 - exact hash match."""
    exp = EXPECTED["q5"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q5 (Weekly OHLC Q4)")


def validate_q6(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q6: AAPL close > open by 5% - exact hash match."""
    exp = EXPECTED["q6"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q6 (Close>Open 5%)")


def validate_q7(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q7: Last 10 AAPL trading days - exact hash match."""
    exp = EXPECTED["q7"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q7 (Last 10 days)")


def validate_q8(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q8: Monthly first/last prices 2023 - exact hash match."""
    exp = EXPECTED["q8"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q8 (Monthly prices)")


def validate_q9(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q9: AAPL up GOOGL down days - exact hash match."""
    exp = EXPECTED["q9"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q9 (AAPL up GOOGL down)")


def validate_q10(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q10: Full year AAPL 2023 data - exact hash match."""
    exp = EXPECTED["q10"]
    return _check_data_hash(response, exp["data_hash"], exp["row_count"], "Q10 (Full year AAPL)")


# ============================================================================
# DATA ANALYSIS VALIDATORS (Q11-Q20)
# ============================================================================

import numpy as np


def validate_q11(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q11: 30-day rolling volatility for AAPL 2023."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q11: Could not extract JSON dict")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict) or "avg_volatility" not in stats:
        return ValidatorResult(False, "Q11: Missing 'avg_volatility'")
    val = stats["avg_volatility"]
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ValidatorResult(False, "Q11: avg_volatility must be a valid number")
    if val <= 0 or val > 2:
        return ValidatorResult(False, f"Q11: avg_volatility {val:.4f} outside range (0, 2)")
    expected = EXPECTED["q11"]["avg_volatility"]
    if np.isclose(val, expected, rtol=0.15):
        return ValidatorResult(True, f"Q11: avg_volatility = {val:.4f} (expected ~{expected:.4f})")
    return ValidatorResult(False, f"Q11: avg_volatility {val:.4f} differs from expected {expected:.4f}")


def validate_q12(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q12 (FC): Count GOOGL days with volume > 100M."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q12: Could not extract JSON dict")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict) or "count" not in stats:
        return ValidatorResult(False, "Q12: Missing 'count'")
    val = int(stats["count"])
    expected = 4  # Pre-computed: query_by_volume_threshold('GOOGL', 100_000_000) returns 4 rows
    if val == expected:
        return ValidatorResult(True, f"Q12: count = {val}")
    return ValidatorResult(False, f"Q12: count {val} differs from expected {expected}")


def validate_q13(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q13: Sharpe ratio for GOOGL 2023."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q13: Could not extract JSON")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict) or "sharpe_ratio" not in stats:
        return ValidatorResult(False, "Q13: Missing 'sharpe_ratio'")
    val = stats["sharpe_ratio"]
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ValidatorResult(False, "Q13: sharpe_ratio must be a valid number")
    if val < -5 or val > 5:
        return ValidatorResult(False, f"Q13: sharpe_ratio {val:.4f} outside range")
    expected = EXPECTED["q13"]["sharpe_ratio"]
    if np.isclose(val, expected, rtol=0.2):
        return ValidatorResult(True, f"Q13: sharpe_ratio = {val:.4f} (expected ~{expected:.4f})")
    return ValidatorResult(False, f"Q13: sharpe_ratio {val:.4f} differs from expected {expected:.4f}")


def validate_q14(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q14: Maximum drawdown for AAPL 2020-2023."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q14: Could not extract JSON")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict) or "max_drawdown" not in stats:
        return ValidatorResult(False, "Q14: Missing 'max_drawdown'")
    val = stats["max_drawdown"]
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ValidatorResult(False, "Q14: max_drawdown must be a valid number")
    abs_val = abs(val)
    if abs_val > 100:
        return ValidatorResult(False, f"Q14: max_drawdown {val} outside range")
    expected = abs(EXPECTED["q14"]["max_drawdown"])
    if np.isclose(abs_val, expected, rtol=0.1):
        return ValidatorResult(True, f"Q14: max_drawdown = {val:.2f}% (expected ~{-expected:.2f}%)")
    return ValidatorResult(False, f"Q14: max_drawdown {val:.2f}% differs from expected {-expected:.2f}%")


def validate_q15(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q15: Monthly return statistics for AAPL 2023 (mean, std, min, max)."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q15: Could not extract JSON")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q15: statistics must be a dict")
    required = ["mean", "std", "min", "max"]
    missing = [k for k in required if k not in stats]
    if missing:
        return ValidatorResult(False, f"Q15: Missing keys: {missing}")
    for key in required:
        if not isinstance(stats[key], (int, float)) or np.isnan(stats[key]):
            return ValidatorResult(False, f"Q15: {key} must be a valid number")
    # Compare against expected values
    exp = EXPECTED["q15"]
    for key in required:
        if not np.isclose(stats[key], exp[key], rtol=0.2):
            return ValidatorResult(False, f"Q15: {key}={stats[key]:.4f} differs from expected {exp[key]:.4f}")
    return ValidatorResult(True, f"Q15: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")


def validate_q16(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q16: Beta of AAPL relative to GOOGL 2023."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q16: Could not extract JSON")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict) or "beta" not in stats:
        return ValidatorResult(False, "Q16: Missing 'beta'")
    val = stats["beta"]
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ValidatorResult(False, "Q16: beta must be a valid number")
    expected = EXPECTED["q16"]["beta"]
    if np.isclose(val, expected, rtol=0.2):
        return ValidatorResult(True, f"Q16: beta = {val:.4f}")
    return ValidatorResult(False, f"Q16: beta {val:.4f} differs from expected {expected:.4f}")


def validate_q17(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q17 (FC): Average close of top 5 AAPL volume days."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q17: Could not extract JSON")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict) or "avg_close" not in stats:
        return ValidatorResult(False, "Q17: Missing 'avg_close'")
    val = float(stats["avg_close"])
    expected = 70.62  # Pre-computed: query_top_volume_days('AAPL', 5)['Close'].mean()
    if np.isclose(val, expected, rtol=0.05):
        return ValidatorResult(True, f"Q17: avg_close = {val:.2f}")
    return ValidatorResult(False, f"Q17: avg_close {val:.2f} differs from expected {expected:.2f}")


def validate_q18(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q18 (FC): Max weekly close for AAPL Q4 2023."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q18: Could not extract JSON")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict) or "max_close" not in stats:
        return ValidatorResult(False, "Q18: Missing 'max_close'")
    val = float(stats["max_close"])
    expected = 196.39  # Pre-computed: query_weekly_ohlc('AAPL', '2023-10-01', '2023-12-31')['Close'].max()
    if np.isclose(val, expected, rtol=0.02):
        return ValidatorResult(True, f"Q18: max_close = {val:.2f}")
    return ValidatorResult(False, f"Q18: max_close {val:.2f} differs from expected {expected:.2f}")


def validate_q19(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q19: Average daily trading range by month for AAPL 2023."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q19: Could not extract JSON")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q19: statistics must be a dict")
    # Look for monthly keys (pattern: 2023-XX) - check both direct and nested
    monthly_data = {}
    for key, val in stats.items():
        if isinstance(key, str) and key.startswith("2023-") and len(key) == 7:
            monthly_data[key] = val
        elif isinstance(val, dict):  # Handle nested structure
            for k, v in val.items():
                if isinstance(k, str) and k.startswith("2023-") and len(k) == 7:
                    monthly_data[k] = v
    if len(monthly_data) < 10:
        return ValidatorResult(False, f"Q19: Expected ~12 month entries, got {len(monthly_data)}")
    # Check values are reasonable and compare average
    values = []
    for key, val in monthly_data.items():
        if isinstance(val, (int, float)):
            if val < 0 or val > 50:
                return ValidatorResult(False, f"Q19: Range {val} for {key} outside reasonable bounds")
            values.append(val)
    if values:
        avg = sum(values) / len(values)
        exp_avg = EXPECTED["q19"]["avg_range"]
        if not np.isclose(avg, exp_avg, rtol=0.2):
            return ValidatorResult(False, f"Q19: avg range {avg:.2f} differs from expected {exp_avg:.2f}")
    return ValidatorResult(True, f"Q19: Monthly ranges for {len(monthly_data)} months")


def validate_q20(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q20: SMA crossover analysis for AAPL 2023."""
    result = extract_json_from_response(response)
    if not isinstance(result, dict):
        return ValidatorResult(False, "Q20: Could not extract JSON")
    stats = result.get("statistics", result)
    if not isinstance(stats, dict) or "crossover_count" not in stats:
        return ValidatorResult(False, "Q20: Missing 'crossover_count'")
    val = int(stats["crossover_count"])
    expected = EXPECTED["q20"]["crossover_count"]
    if abs(val - expected) <= 2:
        return ValidatorResult(True, f"Q20: crossover_count = {val}")
    return ValidatorResult(False, f"Q20: crossover_count {val} differs from expected {expected}")


# ============================================================================
# DATA VISUALIZATION VALIDATORS (Q21-Q30)
# ============================================================================

def validate_q21(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q21: Line chart for AAPL closing prices 2023."""
    result = _check_chart_type(response, "line", "Q21 (AAPL line)", min_series=1)
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    expected_count = EXPECTED["q21"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 5:
        return ValidatorResult(False, f"Q21: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q21: Line chart with {len(series_data)} data points")


def validate_q22(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q22: Candlestick chart for GOOGL Q4 2023."""
    result = _check_chart_type(response, "candlestick", "Q22 (GOOGL candlestick)")
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    expected_count = EXPECTED["q22"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 3:
        return ValidatorResult(False, f"Q22: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q22: Candlestick chart with {len(series_data)} data points")


def validate_q23(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q23: Normalized comparison line chart."""
    result = _check_chart_type(response, "line", "Q23 (Comparison)", min_series=2)
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    expected_count = EXPECTED["q23"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 5:
        return ValidatorResult(False, f"Q23: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q23: Comparison chart with {len(config['series'])} series, {len(series_data)} points")


def validate_q24(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q24: Bar chart for AAPL volume."""
    result = _check_chart_type(response, "bar", "Q24 (Volume bar)")
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    expected_count = EXPECTED["q24"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 5:
        return ValidatorResult(False, f"Q24: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q24: Volume bar chart with {len(series_data)} data points")


def validate_q25(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q25: 30-day rolling correlation chart."""
    result = _check_chart_type(response, "line", "Q25 (Correlation)")
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    series_data = config["series"][0].get("data", [])
    if series_data:
        values = []
        for d in series_data:
            if isinstance(d, (int, float)):
                values.append(d)
            elif isinstance(d, dict) and "value" in d:
                values.append(d["value"])
        if values:
            min_val, max_val = min(values), max(values)
            if min_val < -1.1 or max_val > 1.1:
                return ValidatorResult(False, f"Q25: Correlation values out of range [{min_val:.2f}, {max_val:.2f}]")
    return ValidatorResult(True, f"Q25: Correlation chart with {len(series_data)} data points")


def validate_q26(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q26: Heatmap of monthly returns."""
    result = _check_chart_type(response, "heatmap", "Q26 (Heatmap)")
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    expected_count = EXPECTED["q26"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 5:
        return ValidatorResult(False, f"Q26: Expected ~{expected_count} data points (4yr*12mo), got {len(series_data)}")
    return ValidatorResult(True, f"Q26: Heatmap with {len(series_data)} data points")


def validate_q27(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q27: Scatter plot of returns."""
    result = _check_chart_type(response, "scatter", "Q27 (Scatter)")
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    expected_count = EXPECTED["q27"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 10:
        return ValidatorResult(False, f"Q27: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q27: Scatter plot with {len(series_data)} data points")


def validate_q28(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q28: Bollinger Bands chart."""
    expected_series = EXPECTED["q28"]["series_count"]
    result = _check_chart_type(response, "line", "Q28 (Bollinger)", min_series=expected_series)
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    return ValidatorResult(True, f"Q28: Bollinger Bands with {len(config['series'])} series")


def validate_q29(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q29: Cumulative returns comparison."""
    result = _check_chart_type(response, "line", "Q29 (Cumulative)", min_series=2)
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    exp = EXPECTED["q29"]
    for series in config["series"]:
        data = series.get("data", [])
        if not data:
            continue
        last_val = data[-1] if isinstance(data[-1], (int, float)) else data[-1].get("value", 0)
        name = series.get("name", "").lower()
        if "aapl" in name:
            if abs(last_val - exp["aapl_final"]) > 10:
                return ValidatorResult(False, f"Q29: AAPL final value {last_val:.1f}% differs from expected {exp['aapl_final']:.1f}%")
        elif "googl" in name:
            if abs(last_val - exp["googl_final"]) > 10:
                return ValidatorResult(False, f"Q29: GOOGL final value {last_val:.1f}% differs from expected {exp['googl_final']:.1f}%")
    return ValidatorResult(True, f"Q29: Cumulative returns chart with {len(config['series'])} series")


def validate_q30(response: str, runtime, turn, actual_calls) -> ValidatorResult:
    """Q30: Pie chart of positive vs negative days."""
    result = _check_chart_type(response, "pie", "Q30 (Pie)")
    if not result.success:
        return result
    json_result = extract_json_from_response(response)
    config = json_result.get("chart_config", json_result)
    series_data = config["series"][0].get("data", [])
    if len(series_data) != 2:
        return ValidatorResult(False, f"Q30: Expected 2 pie segments, got {len(series_data)}")
    exp = EXPECTED["q30"]
    expected_pos, expected_neg = exp["positive_days"], exp["negative_days"]
    values = []
    for d in series_data:
        if isinstance(d, dict):
            values.append(d.get("value", 0))
        else:
            values.append(d)
    total = sum(values)
    if abs(total - (expected_pos + expected_neg)) > 5:
        return ValidatorResult(False, f"Q30: Total {total} differs from expected {expected_pos + expected_neg}")
    return ValidatorResult(True, f"Q30: Pie chart with segments summing to {total} days")


# Export all validators
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
    "validate_q16": validate_q16,
    "validate_q17": validate_q17,
    "validate_q18": validate_q18,
    "validate_q19": validate_q19,
    "validate_q20": validate_q20,
    "validate_q21": validate_q21,
    "validate_q22": validate_q22,
    "validate_q23": validate_q23,
    "validate_q24": validate_q24,
    "validate_q25": validate_q25,
    "validate_q26": validate_q26,
    "validate_q27": validate_q27,
    "validate_q28": validate_q28,
    "validate_q29": validate_q29,
    "validate_q30": validate_q30,
}


# ============================================================================
# EXPORTS - Only query tools, NO analysis/visualization tools
# ============================================================================

tools = [
    query_by_date_range,
    query_by_volume_threshold,
    query_top_volume_days,
    query_price_change_days,
    query_last_n_days,
    query_weekly_ohlc,
    query_monthly_first_last,
    query_52week_high_days,
    query_cross_stock_condition,
]
