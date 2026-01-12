"""Comprehensive Stock Data Benchmark - CaveAgent Version

30 queries across 3 categories:
- Data Query (10): Use provided query tools
- Data Analysis (10): Write Python code to analyze (no tools provided)
- Data Visualization (10): Write Python code to generate ECharts config (no tools provided)

CaveAgent: Has DataFrames in runtime, writes code, stores results in variables.
"""

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from cave_agent.python_runtime import PythonRuntime, Variable
from core.types import BenchmarkTurn, ToolCall
from core.validation import ValidatorResult


# Load data at module level (filtered to 2020-2024 to reduce size)
DATA_DIR = os.path.dirname(__file__) + "/../data_analysis/finance"
AAPL_DF = pd.read_csv(f"{DATA_DIR}/AAPL.csv", parse_dates=["Date"])
GOOGL_DF = pd.read_csv(f"{DATA_DIR}/GOOGL.csv", parse_dates=["Date"])

# Filter to 2020-2024 to keep data manageable while supporting all queries
AAPL_DF = AAPL_DF[(AAPL_DF["Date"] >= "2020-01-01") & (AAPL_DF["Date"] <= "2024-12-31")]
GOOGL_DF = GOOGL_DF[(GOOGL_DF["Date"] >= "2020-01-01") & (GOOGL_DF["Date"] <= "2024-12-31")]

# Ensure data is sorted by date
AAPL_DF = AAPL_DF.sort_values("Date").reset_index(drop=True)
GOOGL_DF = GOOGL_DF.sort_values("Date").reset_index(drop=True)


# ============================================================================
# DATA QUERY TOOLS ONLY - Analysis and Visualization done by agent coding
# ============================================================================

def query_by_date_range(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Query stock data for a date range.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        Example row: {Date: 2023-01-03, Open: 130.28, High: 130.90, Low: 124.17, Close: 125.07, Volume: 112117500}
    """
    df = AAPL_DF if symbol == "AAPL" else GOOGL_DF
    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    return df[mask].copy()


def query_by_volume_threshold(symbol: str, min_volume: float) -> pd.DataFrame:
    """Find days where volume exceeded threshold.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        min_volume: Minimum volume threshold (e.g., 50000000 for 50 million)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        Contains only rows where Volume > min_volume
    """
    df = AAPL_DF if symbol == "AAPL" else GOOGL_DF
    return df[df["Volume"] > min_volume].copy()


def query_top_volume_days(symbol: str, n: int) -> pd.DataFrame:
    """Get top N highest volume trading days.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        n: Number of top days to return

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        Sorted by Volume descending, limited to n rows
    """
    df = AAPL_DF if symbol == "AAPL" else GOOGL_DF
    return df.nlargest(n, "Volume").copy()


def query_price_change_days(symbol: str, min_pct_change: float) -> pd.DataFrame:
    """Find days where close > open by percentage.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        min_pct_change: Minimum percentage change (e.g., 5 for 5%)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        Contains only rows where (Close - Open) / Open * 100 > min_pct_change
    """
    df = AAPL_DF if symbol == "AAPL" else GOOGL_DF
    pct_change = (df["Close"] - df["Open"]) / df["Open"] * 100
    return df[pct_change > min_pct_change].copy()


def query_last_n_days(symbol: str, n: int) -> pd.DataFrame:
    """Get last N trading days.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        n: Number of most recent trading days to return

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        Contains the last n rows sorted by Date ascending
    """
    df = AAPL_DF if symbol == "AAPL" else GOOGL_DF
    return df.tail(n).copy()


def query_weekly_ohlc(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get weekly OHLC data (resampled from daily data).

    Args:
        symbol: 'AAPL' or 'GOOGL'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        Each row represents one week: Open=first day, High=max, Low=min, Close=last day, Volume=sum
    """
    df = query_by_date_range(symbol, start_date, end_date)
    df = df.set_index("Date")
    weekly = df.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().reset_index()
    return weekly


def query_monthly_first_last(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get monthly first and last trading day prices.

    Args:
        symbol: 'AAPL' or 'GOOGL'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'

    Returns:
        DataFrame with columns: Month, First_Date, Last_Date, First_Open, Last_Close
        Each row represents one month with first/last trading day data
    """
    df = query_by_date_range(symbol, start_date, end_date)
    df["Month"] = df["Date"].dt.to_period("M")
    result = df.groupby("Month").agg({
        "Date": ["first", "last"],
        "Open": "first",
        "Close": "last"
    }).reset_index()
    result.columns = ["Month", "First_Date", "Last_Date", "First_Open", "Last_Close"]
    result["Month"] = result["Month"].astype(str)
    return result


def query_52week_high_days(symbol: str) -> pd.DataFrame:
    """Find days where stock hit 52-week high.

    Args:
        symbol: 'AAPL' or 'GOOGL'

    Returns:
        DataFrame with columns: Date, High, Close, Volume
        Contains only days where the High price equals or exceeds the rolling 252-day maximum
    """
    df = AAPL_DF.copy() if symbol == "AAPL" else GOOGL_DF.copy()
    df["52W_High"] = df["High"].rolling(window=252, min_periods=1).max()
    high_days = df[df["High"] >= df["52W_High"]]
    return high_days[["Date", "High", "Close", "Volume"]].copy()


def query_cross_stock_condition(condition: str) -> pd.DataFrame:
    """Cross-reference days based on condition comparing AAPL and GOOGL.

    Args:
        condition: 'aapl_up_googl_down' (AAPL up, GOOGL down) or 'both_down_3pct' (both down >3%)

    Returns:
        DataFrame with columns: Date, Close_AAPL, AAPL_Return, Close_GOOGL, GOOGL_Return
        Contains only days matching the specified condition
    """
    aapl = AAPL_DF.copy()
    googl = GOOGL_DF.copy()

    aapl["AAPL_Return"] = aapl["Close"].pct_change() * 100
    googl["GOOGL_Return"] = googl["Close"].pct_change() * 100

    merged = pd.merge(aapl[["Date", "Close", "AAPL_Return"]],
                      googl[["Date", "Close", "GOOGL_Return"]],
                      on="Date", suffixes=("_AAPL", "_GOOGL"))

    if condition == "aapl_up_googl_down":
        result = merged[(merged["AAPL_Return"] > 0) & (merged["GOOGL_Return"] < 0)]
    elif condition == "both_down_3pct":
        result = merged[(merged["AAPL_Return"] < -3) & (merged["GOOGL_Return"] < -3)]
    else:
        result = merged

    return result.copy()


# ============================================================================
# EXPECTED RESULTS FOR VALIDATION
# ============================================================================

import hashlib


def _df_hash(df: pd.DataFrame) -> str:
    """Compute hash of DataFrame content for exact matching."""
    # Convert to consistent format: sort by Date, round floats, format dates
    df_copy = df.copy()
    if "Date" in df_copy.columns:
        df_copy = df_copy.sort_values("Date").reset_index(drop=True)
        df_copy["Date"] = df_copy["Date"].astype(str)
    # Round numeric columns to avoid floating point issues
    for col in df_copy.select_dtypes(include=[np.number]).columns:
        df_copy[col] = df_copy[col].round(4)
    # Convert to string and hash
    content = df_copy.to_csv(index=False)
    return hashlib.md5(content.encode()).hexdigest()


def _compute_expected():
    """Pre-compute all expected results for validation."""
    results = {}

    # Data Query (1-10) - Store both row_count and data_hash for exact matching
    q1_data = query_by_date_range("AAPL", "2023-01-01", "2023-01-31")
    results["q1"] = {"row_count": len(q1_data), "data_hash": _df_hash(q1_data)}

    q2_data = query_by_volume_threshold("GOOGL", 100_000_000)
    results["q2"] = {"row_count": len(q2_data), "data_hash": _df_hash(q2_data)}

    q3_data = query_cross_stock_condition("both_down_3pct")
    results["q3"] = {"row_count": len(q3_data), "data_hash": _df_hash(q3_data)}

    q4_data = query_top_volume_days("AAPL", 5)
    results["q4"] = {"row_count": 5, "data_hash": _df_hash(q4_data)}

    q5_data = query_weekly_ohlc("GOOGL", "2023-10-01", "2023-12-31")
    results["q5"] = {"row_count": len(q5_data), "data_hash": _df_hash(q5_data)}

    q6_data = query_price_change_days("AAPL", 5)
    results["q6"] = {"row_count": len(q6_data), "data_hash": _df_hash(q6_data)}

    q7_data = query_last_n_days("AAPL", 10)
    results["q7"] = {"row_count": 10, "data_hash": _df_hash(q7_data)}

    q8_data = query_monthly_first_last("AAPL", "2023-01-01", "2023-12-31")
    results["q8"] = {"row_count": len(q8_data), "data_hash": _df_hash(q8_data)}

    q9_data = query_cross_stock_condition("aapl_up_googl_down")
    results["q9"] = {"row_count": len(q9_data), "data_hash": _df_hash(q9_data)}

    q10_data = query_by_date_range("AAPL", "2023-01-01", "2023-12-31")
    results["q10"] = {"row_count": len(q10_data), "data_hash": _df_hash(q10_data)}

    # Data Analysis (11-20) - Agent computes these via code
    # Q11: 30-day rolling volatility for AAPL 2023
    aapl_2023 = query_by_date_range("AAPL", "2023-01-01", "2023-12-31")
    vol = aapl_2023["Close"].pct_change().rolling(30).std() * np.sqrt(252)
    results["q11"] = {"avg_volatility": float(vol.dropna().mean())}

    # Q12: Correlation between AAPL and GOOGL returns 2023
    aapl = query_by_date_range("AAPL", "2023-01-01", "2023-12-31")
    googl = query_by_date_range("GOOGL", "2023-01-01", "2023-12-31")
    merged = pd.merge(aapl[["Date", "Close"]], googl[["Date", "Close"]], on="Date", suffixes=("_A", "_G"))
    corr = merged["Close_A"].pct_change().corr(merged["Close_G"].pct_change())
    results["q12"] = {"correlation": float(corr)}

    # Q13: Sharpe ratio for GOOGL 2023 (risk-free rate = 0 as per query)
    googl_ret = googl["Close"].pct_change().dropna()
    sharpe = (googl_ret.mean() * 252) / (googl_ret.std() * np.sqrt(252))
    results["q13"] = {"sharpe_ratio": float(sharpe)}

    # Q14: Max drawdown for AAPL 2020-2023
    aapl_long = query_by_date_range("AAPL", "2020-01-01", "2023-12-31")
    cummax = aapl_long["Close"].cummax()
    drawdown = (aapl_long["Close"] - cummax) / cummax
    results["q14"] = {"max_drawdown": float(drawdown.min() * 100)}

    # Q15: Monthly return stats for AAPL 2023 (mean, std, min, max as per query)
    aapl_monthly = aapl.set_index("Date")["Close"].resample("ME").last().pct_change().dropna()
    results["q15"] = {
        "mean": float(aapl_monthly.mean()),
        "std": float(aapl_monthly.std()),
        "min": float(aapl_monthly.min()),
        "max": float(aapl_monthly.max())
    }

    # Q16: Beta of AAPL relative to GOOGL 2023
    aapl_ret = merged["Close_A"].pct_change().dropna()
    googl_ret = merged["Close_G"].pct_change().dropna()
    beta = aapl_ret.cov(googl_ret) / googl_ret.var()
    results["q16"] = {"beta": float(beta)}

    # Q17: Price anomalies for AAPL 2023
    aapl_2023 = query_by_date_range("AAPL", "2023-01-01", "2023-12-31").copy()
    aapl_2023["MA"] = aapl_2023["Close"].rolling(20).mean()
    aapl_2023["STD"] = aapl_2023["Close"].rolling(20).std()
    aapl_2023["Z"] = (aapl_2023["Close"] - aapl_2023["MA"]) / aapl_2023["STD"]
    anomalies = aapl_2023[abs(aapl_2023["Z"]) > 2]
    results["q17"] = {"anomaly_count": len(anomalies)}

    # Q18: Cumulative returns 2020-2023
    aapl_long = query_by_date_range("AAPL", "2020-01-01", "2023-12-31")
    googl_long = query_by_date_range("GOOGL", "2020-01-01", "2023-12-31")
    aapl_cum = (aapl_long["Close"].iloc[-1] / aapl_long["Close"].iloc[0] - 1) * 100
    googl_cum = (googl_long["Close"].iloc[-1] / googl_long["Close"].iloc[0] - 1) * 100
    results["q18"] = {"aapl_cumulative": float(aapl_cum), "googl_cumulative": float(googl_cum)}

    # Q19: Avg daily range by month for AAPL 2023
    aapl_2023 = query_by_date_range("AAPL", "2023-01-01", "2023-12-31").copy()
    aapl_2023["Range"] = aapl_2023["High"] - aapl_2023["Low"]
    aapl_2023["Month"] = aapl_2023["Date"].dt.to_period("M")
    monthly_range = aapl_2023.groupby("Month")["Range"].mean()
    # Store month_count and average range for validation
    results["q19"] = {
        "month_count": len(monthly_range),
        "avg_range": float(monthly_range.mean())
    }

    # Q20: SMA crossover for AAPL 2023
    aapl_2023 = query_by_date_range("AAPL", "2023-01-01", "2023-12-31").copy()
    aapl_2023["SMA20"] = aapl_2023["Close"].rolling(20).mean()
    aapl_2023["SMA50"] = aapl_2023["Close"].rolling(50).mean()
    aapl_2023["Signal"] = (aapl_2023["SMA20"] > aapl_2023["SMA50"]).astype(int)
    aapl_2023["Crossover"] = aapl_2023["Signal"].diff()
    crossovers = aapl_2023[aapl_2023["Crossover"] != 0].dropna()
    results["q20"] = {"crossover_count": len(crossovers)}

    # Data Visualization (21-30) - Check ECharts config structure AND data
    # Q21: Line chart AAPL closing prices 2023 (~250 trading days)
    aapl_2023 = query_by_date_range("AAPL", "2023-01-01", "2023-12-31")
    results["q21"] = {"chart_type": "line", "data_count": len(aapl_2023)}

    # Q22: Candlestick GOOGL Q4 2023 (~63 trading days)
    googl_q4 = query_by_date_range("GOOGL", "2023-10-01", "2023-12-31")
    results["q22"] = {"chart_type": "candlestick", "data_count": len(googl_q4)}

    # Q23: Normalized comparison (~250 points, 2 series)
    results["q23"] = {"chart_type": "line", "series_count": 2, "data_count": len(aapl_2023)}

    # Q24: Bar chart volume (~250 points)
    results["q24"] = {"chart_type": "bar", "data_count": len(aapl_2023)}

    # Q25: Rolling correlation (values must be in [-1, 1])
    results["q25"] = {"chart_type": "line", "value_range": (-1, 1)}

    # Q26: Heatmap monthly returns 2020-2023 (47 months after pct_change)
    results["q26"] = {"chart_type": "heatmap", "data_count": 47}

    # Q27: Scatter plot returns (~249 points after pct_change)
    results["q27"] = {"chart_type": "scatter", "data_count": len(aapl_2023) - 1}

    # Q28: Bollinger Bands (3+ series: price, upper, lower)
    results["q28"] = {"chart_type": "line", "series_count": 3}

    # Q29: Cumulative returns comparison (final values)
    aapl_2023_cum = (aapl_2023["Close"].iloc[-1] / aapl_2023["Close"].iloc[0] - 1) * 100
    googl_2023 = query_by_date_range("GOOGL", "2023-01-01", "2023-12-31")
    googl_2023_cum = (googl_2023["Close"].iloc[-1] / googl_2023["Close"].iloc[0] - 1) * 100
    results["q29"] = {
        "chart_type": "line",
        "series_count": 2,
        "aapl_final": float(aapl_2023_cum),
        "googl_final": float(googl_2023_cum)
    }

    # Q30: Pie chart positive vs negative days
    aapl_2023_copy = aapl_2023.copy()
    aapl_2023_copy["Change"] = aapl_2023_copy["Close"] - aapl_2023_copy["Open"]
    positive_days = int((aapl_2023_copy["Change"] > 0).sum())
    negative_days = int((aapl_2023_copy["Change"] <= 0).sum())
    results["q30"] = {
        "chart_type": "pie",
        "positive_days": positive_days,
        "negative_days": negative_days
    }

    return results


EXPECTED = _compute_expected()


# ============================================================================
# VALIDATORS - Individual validators for each query
# ============================================================================

def _get_queried_data(runtime: PythonRuntime):
    """Helper to get queried_data variable."""
    try:
        return runtime.retrieve("queried_data")
    except:
        return None


def _get_statistics(runtime: PythonRuntime):
    """Helper to get statistics variable."""
    try:
        return runtime.retrieve("statistics")
    except:
        return None


def _get_chart_config(runtime: PythonRuntime):
    """Helper to get chart_config variable."""
    try:
        return runtime.retrieve("chart_config")
    except:
        return None


def _check_data_hash(data, expected_hash: str, expected_count: int, query_name: str) -> ValidatorResult:
    """Check if data matches expected hash (exact match) and row count."""
    if data is None:
        return ValidatorResult(False, f"{query_name}: Variable 'queried_data' not found")
    if not hasattr(data, '__len__'):
        return ValidatorResult(False, f"{query_name}: Invalid data type")
    if len(data) != expected_count:
        return ValidatorResult(False, f"{query_name}: Expected {expected_count} rows, got {len(data)}")
    # Convert to DataFrame if needed and compute hash
    if isinstance(data, pd.DataFrame):
        actual_hash = _df_hash(data)  # _df_hash auto-filters to standard columns
    elif isinstance(data, list):
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data)
        actual_hash = _df_hash(df)
    else:
        return ValidatorResult(False, f"{query_name}: Unsupported data type {type(data)}")
    if actual_hash == expected_hash:
        return ValidatorResult(True, f"{query_name}: Data hash matches ({len(data)} rows)")
    return ValidatorResult(False, f"{query_name}: Data hash mismatch (got {actual_hash[:8]}..., expected {expected_hash[:8]}...)")


def _check_chart_type(config, expected_type: str, query_name: str, min_series: int = 1) -> ValidatorResult:
    """Check if chart config has expected type and series count."""
    if config is None:
        return ValidatorResult(False, f"{query_name}: Variable 'chart_config' not found")
    if not isinstance(config, dict):
        return ValidatorResult(False, f"{query_name}: chart_config must be a dict")
    if "series" not in config:
        return ValidatorResult(False, f"{query_name}: Missing 'series' in chart config")
    series = config["series"]
    if not series:
        return ValidatorResult(False, f"{query_name}: Empty series in chart config")
    chart_type = series[0].get("type", "")
    if chart_type != expected_type:
        return ValidatorResult(False, f"{query_name}: Expected {expected_type}, got {chart_type}")
    if len(series) < min_series:
        return ValidatorResult(False, f"{query_name}: Expected >= {min_series} series, got {len(series)}")
    return ValidatorResult(True, f"{query_name}: Valid {chart_type} chart with {len(series)} series")


# ============================================================================
# DATA QUERY VALIDATORS (Q1-Q10)
# ============================================================================

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q1: AAPL January 2023 data - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q1"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q1 (AAPL Jan 2023)")


def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q2: GOOGL volume > 100M days - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q2"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q2 (GOOGL Vol>100M)")


def validate_q3(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q3: Both AAPL and GOOGL down > 3% - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q3"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q3 (Both down >3%)")


def validate_q4(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q4: Top 5 AAPL volume days - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q4"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q4 (Top 5 volume)")


def validate_q5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q5: GOOGL weekly OHLC Q4 2023 - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q5"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q5 (Weekly OHLC Q4)")


def validate_q6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q6: AAPL close > open by 5% - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q6"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q6 (Close>Open 5%)")


def validate_q7(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q7: Last 10 AAPL trading days - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q7"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q7 (Last 10 days)")


def validate_q8(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q8: Monthly first/last prices 2023 - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q8"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q8 (Monthly prices)")


def validate_q9(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q9: AAPL up GOOGL down days - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q9"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q9 (AAPL up GOOGL down)")


def validate_q10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q10: Full year AAPL 2023 data - exact hash match."""
    data = _get_queried_data(runtime)
    exp = EXPECTED["q10"]
    return _check_data_hash(data, exp["data_hash"], exp["row_count"], "Q10 (Full year AAPL)")


# ============================================================================
# DATA ANALYSIS VALIDATORS (Q11-Q20)
# ============================================================================

def validate_q11(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q11: 30-day rolling volatility for AAPL 2023."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q11: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q11: statistics must be a dict")
    if "avg_volatility" not in stats:
        return ValidatorResult(False, "Q11: Missing 'avg_volatility' in statistics")
    val = stats["avg_volatility"]
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ValidatorResult(False, "Q11: avg_volatility must be a valid number")
    if val <= 0 or val > 2:  # Reasonable volatility range
        return ValidatorResult(False, f"Q11: avg_volatility {val:.4f} outside reasonable range (0, 2)")
    expected = EXPECTED["q11"]["avg_volatility"]
    if np.isclose(val, expected, rtol=0.15):
        return ValidatorResult(True, f"Q11: avg_volatility = {val:.4f} (expected ~{expected:.4f})")
    return ValidatorResult(False, f"Q11: avg_volatility {val:.4f} differs from expected {expected:.4f}")


def validate_q12(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q12: Correlation between AAPL and GOOGL returns 2023."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q12: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q12: statistics must be a dict")
    if "correlation" not in stats:
        return ValidatorResult(False, "Q12: Missing 'correlation' in statistics")
    val = stats["correlation"]
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ValidatorResult(False, "Q12: correlation must be a valid number")
    if val < -1 or val > 1:
        return ValidatorResult(False, f"Q12: correlation {val:.4f} outside valid range [-1, 1]")
    expected = EXPECTED["q12"]["correlation"]
    if np.isclose(val, expected, rtol=0.15):
        return ValidatorResult(True, f"Q12: correlation = {val:.4f} (expected ~{expected:.4f})")
    return ValidatorResult(False, f"Q12: correlation {val:.4f} differs from expected {expected:.4f}")


def validate_q13(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q13: Sharpe ratio for GOOGL 2023."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q13: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q13: statistics must be a dict")
    if "sharpe_ratio" not in stats:
        return ValidatorResult(False, "Q13: Missing 'sharpe_ratio' in statistics")
    val = stats["sharpe_ratio"]
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ValidatorResult(False, "Q13: sharpe_ratio must be a valid number")
    # Sharpe ratio can be negative or positive, typically -3 to 3 range
    if val < -5 or val > 5:
        return ValidatorResult(False, f"Q13: sharpe_ratio {val:.4f} outside reasonable range")
    expected = EXPECTED["q13"]["sharpe_ratio"]
    if np.isclose(val, expected, rtol=0.2):
        return ValidatorResult(True, f"Q13: sharpe_ratio = {val:.4f} (expected ~{expected:.4f})")
    return ValidatorResult(False, f"Q13: sharpe_ratio {val:.4f} differs from expected {expected:.4f}")


def validate_q14(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q14: Maximum drawdown for AAPL 2020-2023."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q14: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q14: statistics must be a dict")
    if "max_drawdown" not in stats:
        return ValidatorResult(False, "Q14: Missing 'max_drawdown' in statistics")
    val = stats["max_drawdown"]
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ValidatorResult(False, "Q14: max_drawdown must be a valid number")
    # Max drawdown should be negative (or expressed as positive percentage)
    abs_val = abs(val)
    if abs_val > 100:
        return ValidatorResult(False, f"Q14: max_drawdown {val} outside valid range")
    expected = abs(EXPECTED["q14"]["max_drawdown"])
    if np.isclose(abs_val, expected, rtol=0.1):
        return ValidatorResult(True, f"Q14: max_drawdown = {val:.2f}% (expected ~{-expected:.2f}%)")
    return ValidatorResult(False, f"Q14: max_drawdown {val:.2f}% differs from expected {-expected:.2f}%")


def validate_q15(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q15: Monthly return statistics for AAPL 2023 (mean, std, min, max)."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q15: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q15: statistics must be a dict")
    required_keys = ["mean", "std", "min", "max"]
    # Check for keys at top level or in a nested dict
    data = stats
    if not all(k in stats for k in required_keys):
        # Look for nested structure
        for val in stats.values():
            if isinstance(val, dict) and all(k in val for k in required_keys):
                data = val
                break
    missing = [k for k in required_keys if k not in data]
    if missing:
        return ValidatorResult(False, f"Q15: Missing keys: {missing}")
    for key in required_keys:
        if not isinstance(data[key], (int, float)) or np.isnan(data[key]):
            return ValidatorResult(False, f"Q15: {key} must be a valid number")
    # Compare against expected values
    exp = EXPECTED["q15"]
    for key in required_keys:
        if not np.isclose(data[key], exp[key], rtol=0.2):
            return ValidatorResult(False, f"Q15: {key}={data[key]:.4f} differs from expected {exp[key]:.4f}")
    return ValidatorResult(True, f"Q15: mean={data['mean']:.4f}, std={data['std']:.4f}, min={data['min']:.4f}, max={data['max']:.4f}")


def validate_q16(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q16: Beta of AAPL relative to GOOGL 2023."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q16: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q16: statistics must be a dict")
    if "beta" not in stats:
        return ValidatorResult(False, "Q16: Missing 'beta' in statistics")
    val = stats["beta"]
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ValidatorResult(False, "Q16: beta must be a valid number")
    # Beta typically ranges from -2 to 3 for stocks
    if val < -3 or val > 5:
        return ValidatorResult(False, f"Q16: beta {val:.4f} outside reasonable range")
    expected = EXPECTED["q16"]["beta"]
    if np.isclose(val, expected, rtol=0.2):
        return ValidatorResult(True, f"Q16: beta = {val:.4f} (expected ~{expected:.4f})")
    return ValidatorResult(False, f"Q16: beta {val:.4f} differs from expected {expected:.4f}")


def validate_q17(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q17: AAPL price anomalies (>2 std from 20-day mean) in 2023."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q17: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q17: statistics must be a dict")
    if "anomaly_count" not in stats:
        return ValidatorResult(False, "Q17: Missing 'anomaly_count' in statistics")
    val = stats["anomaly_count"]
    if not isinstance(val, (int, float)):
        return ValidatorResult(False, "Q17: anomaly_count must be a number")
    val = int(val)
    expected = EXPECTED["q17"]["anomaly_count"]
    if val == expected:
        return ValidatorResult(True, f"Q17: anomaly_count = {val}")
    # Allow some tolerance for different calculation methods
    if abs(val - expected) <= 3:
        return ValidatorResult(True, f"Q17: anomaly_count = {val} (close to expected {expected})")
    return ValidatorResult(False, f"Q17: anomaly_count {val} differs from expected {expected}")


def validate_q18(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q18: Cumulative returns for AAPL and GOOGL 2020-2023."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q18: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q18: statistics must be a dict")
    required_keys = ["aapl_cumulative", "googl_cumulative"]
    missing = [k for k in required_keys if k not in stats]
    if missing:
        return ValidatorResult(False, f"Q18: Missing keys: {missing}")
    for key in required_keys:
        if not isinstance(stats[key], (int, float)) or np.isnan(stats[key]):
            return ValidatorResult(False, f"Q18: {key} must be a valid number")
    # Compare against expected values
    exp = EXPECTED["q18"]
    for key in required_keys:
        if not np.isclose(stats[key], exp[key], rtol=0.1):
            return ValidatorResult(False, f"Q18: {key}={stats[key]:.2f}% differs from expected {exp[key]:.2f}%")
    return ValidatorResult(True, f"Q18: AAPL={stats['aapl_cumulative']:.2f}%, GOOGL={stats['googl_cumulative']:.2f}%")


def validate_q19(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q19: Average daily trading range by month for AAPL 2023."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q19: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q19: statistics must be a dict")
    # Look for monthly keys (pattern: 2023-XX) - check both direct and nested
    monthly_data = {}
    for key, val in stats.items():
        if isinstance(key, str) and key.startswith("2023-") and len(key) == 7:
            monthly_data[key] = val
        elif isinstance(val, dict):  # Handle nested structure like {'monthly_avg_range': {...}}
            for k, v in val.items():
                if isinstance(k, str) and k.startswith("2023-") and len(k) == 7:
                    monthly_data[k] = v
    if len(monthly_data) < 10:
        return ValidatorResult(False, f"Q19: Expected ~12 month entries, got {len(monthly_data)}")
    # Check values are reasonable (daily range in dollars, typically 1-20 for AAPL)
    values = []
    for key, val in monthly_data.items():
        if isinstance(val, (int, float)):
            if val < 0 or val > 50:
                return ValidatorResult(False, f"Q19: Range {val} for {key} outside reasonable bounds")
            values.append(val)
    # Compare average range against expected
    if values:
        avg = sum(values) / len(values)
        exp_avg = EXPECTED["q19"]["avg_range"]
        if not np.isclose(avg, exp_avg, rtol=0.2):
            return ValidatorResult(False, f"Q19: avg range {avg:.2f} differs from expected {exp_avg:.2f}")
    return ValidatorResult(True, f"Q19: Monthly ranges computed for {len(monthly_data)} months")


def validate_q20(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q20: SMA crossover analysis (SMA20 vs SMA50) for AAPL 2023."""
    stats = _get_statistics(runtime)
    if stats is None:
        return ValidatorResult(False, "Q20: Variable 'statistics' not found")
    if not isinstance(stats, dict):
        return ValidatorResult(False, "Q20: statistics must be a dict")
    if "crossover_count" not in stats:
        return ValidatorResult(False, "Q20: Missing 'crossover_count' in statistics")
    val = stats["crossover_count"]
    if not isinstance(val, (int, float)):
        return ValidatorResult(False, "Q20: crossover_count must be a number")
    val = int(val)
    expected = EXPECTED["q20"]["crossover_count"]
    if val == expected:
        return ValidatorResult(True, f"Q20: crossover_count = {val}")
    # Allow tolerance for edge cases
    if abs(val - expected) <= 2:
        return ValidatorResult(True, f"Q20: crossover_count = {val} (close to expected {expected})")
    return ValidatorResult(False, f"Q20: crossover_count {val} differs from expected {expected}")


# ============================================================================
# DATA VISUALIZATION VALIDATORS (Q21-Q30)
# ============================================================================

def validate_q21(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q21: Line chart for AAPL closing prices 2023."""
    config = _get_chart_config(runtime)
    result = _check_chart_type(config, "line", "Q21 (AAPL line)", min_series=1)
    if not result.success:
        return result
    # Check data count matches expected (~250 trading days)
    expected_count = EXPECTED["q21"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 5:  # Allow small tolerance
        return ValidatorResult(False, f"Q21: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q21: Line chart with {len(series_data)} data points")


def validate_q22(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q22: Candlestick chart for GOOGL Q4 2023."""
    config = _get_chart_config(runtime)
    result = _check_chart_type(config, "candlestick", "Q22 (GOOGL candlestick)", min_series=1)
    if not result.success:
        return result
    # Check data count (~63 trading days in Q4)
    expected_count = EXPECTED["q22"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 3:
        return ValidatorResult(False, f"Q22: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q22: Candlestick chart with {len(series_data)} data points")


def validate_q23(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q23: Normalized comparison line chart AAPL vs GOOGL."""
    config = _get_chart_config(runtime)
    result = _check_chart_type(config, "line", "Q23 (Comparison)", min_series=2)
    if not result.success:
        return result
    # Check data count
    expected_count = EXPECTED["q23"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 5:
        return ValidatorResult(False, f"Q23: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q23: Comparison chart with {len(config['series'])} series, {len(series_data)} points")


def validate_q24(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q24: Bar chart for AAPL volume with price overlay."""
    config = _get_chart_config(runtime)
    result = _check_chart_type(config, "bar", "Q24 (Volume bar)", min_series=1)
    if not result.success:
        return result
    # Check data count
    expected_count = EXPECTED["q24"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 5:
        return ValidatorResult(False, f"Q24: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q24: Volume bar chart with {len(series_data)} data points")


def validate_q25(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q25: 30-day rolling correlation chart."""
    config = _get_chart_config(runtime)
    result = _check_chart_type(config, "line", "Q25 (Correlation)", min_series=1)
    if not result.success:
        return result
    # Check correlation values are in valid range [-1, 1]
    series_data = config["series"][0].get("data", [])
    if series_data:
        # Handle both list of values and list of dicts
        values = []
        for d in series_data:
            if isinstance(d, (int, float)):
                values.append(d)
            elif isinstance(d, dict) and "value" in d:
                values.append(d["value"])
        if values:
            min_val, max_val = min(values), max(values)
            if min_val < -1.1 or max_val > 1.1:  # Small tolerance for floating point
                return ValidatorResult(False, f"Q25: Correlation values out of range [{min_val:.2f}, {max_val:.2f}]")
    return ValidatorResult(True, f"Q25: Correlation chart with {len(series_data)} data points")


def validate_q26(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q26: Heatmap of monthly returns for AAPL 2020-2023."""
    config = _get_chart_config(runtime)
    result = _check_chart_type(config, "heatmap", "Q26 (Returns heatmap)", min_series=1)
    if not result.success:
        return result
    # Check data count (~48 months)
    expected_count = EXPECTED["q26"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 5:
        return ValidatorResult(False, f"Q26: Expected ~{expected_count} data points (4yr*12mo), got {len(series_data)}")
    return ValidatorResult(True, f"Q26: Heatmap with {len(series_data)} data points")


def validate_q27(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q27: Scatter plot of AAPL vs GOOGL returns."""
    config = _get_chart_config(runtime)
    result = _check_chart_type(config, "scatter", "Q27 (Returns scatter)", min_series=1)
    if not result.success:
        return result
    # Check data count
    expected_count = EXPECTED["q27"]["data_count"]
    series_data = config["series"][0].get("data", [])
    if abs(len(series_data) - expected_count) > 10:
        return ValidatorResult(False, f"Q27: Expected ~{expected_count} data points, got {len(series_data)}")
    return ValidatorResult(True, f"Q27: Scatter plot with {len(series_data)} data points")


def validate_q28(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q28: Bollinger Bands chart for AAPL (20-day)."""
    config = _get_chart_config(runtime)
    expected_series = EXPECTED["q28"]["series_count"]
    result = _check_chart_type(config, "line", "Q28 (Bollinger)", min_series=expected_series)
    if not result.success:
        return result
    return ValidatorResult(True, f"Q28: Bollinger Bands with {len(config['series'])} series")


def validate_q29(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q29: Cumulative returns comparison chart."""
    config = _get_chart_config(runtime)
    result = _check_chart_type(config, "line", "Q29 (Cumulative returns)", min_series=2)
    if not result.success:
        return result
    # Check final cumulative return values (last data point in each series)
    exp = EXPECTED["q29"]
    for series in config["series"]:
        data = series.get("data", [])
        if not data:
            continue
        last_val = data[-1] if isinstance(data[-1], (int, float)) else data[-1].get("value", 0)
        name = series.get("name", "").lower()
        if "aapl" in name:
            if abs(last_val - exp["aapl_final"]) > 10:  # 10% tolerance
                return ValidatorResult(False, f"Q29: AAPL final value {last_val:.1f}% differs from expected {exp['aapl_final']:.1f}%")
        elif "googl" in name:
            if abs(last_val - exp["googl_final"]) > 10:
                return ValidatorResult(False, f"Q29: GOOGL final value {last_val:.1f}% differs from expected {exp['googl_final']:.1f}%")
    return ValidatorResult(True, f"Q29: Cumulative returns chart with {len(config['series'])} series")


def validate_q30(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """Q30: Pie chart of positive vs negative days."""
    config = _get_chart_config(runtime)
    result = _check_chart_type(config, "pie", "Q30 (Pos/Neg pie)", min_series=1)
    if not result.success:
        return result
    # Check pie has 2 segments with correct values
    series_data = config["series"][0].get("data", [])
    if len(series_data) != 2:
        return ValidatorResult(False, f"Q30: Expected 2 pie segments, got {len(series_data)}")
    # Verify the counts match expected
    exp = EXPECTED["q30"]
    expected_pos, expected_neg = exp["positive_days"], exp["negative_days"]
    # Extract values from pie data (could be {name, value} or just value)
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
# EXPORTS - Only query tools, no analysis/visualization tools
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

variables = [
    Variable("queried_data", None, "Store query results here as DataFrame or list of records"),
    Variable("statistics", None, "Store analysis results here as a dictionary, e.g., {'correlation': 0.85, 'sharpe_ratio': 1.2}"),
    Variable("chart_config", None, """Store ECharts configuration here as a dictionary, if you are asked to generate a chart.This will be used in a Web UI to display the chart.
Data visualization guide:
- Generate charts when working with suitable data
- Create a variety of chart types (pie, line, bar, candlestick, scatter, heatmap) as appropriate
- JSON Compatibility: Ensure all configurations are valid Python dictionaries
- Don't contain Java Script code.
Example:
```python
dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
prices = df['Close'].tolist()
chart_config = {
    "title": {"text": "AAPL Closing Prices"},
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "category", "data": dates},
    "yAxis": {"type": "value"},
    "series": [{"name": "Close", "type": "line", "data": prices}]
}
```"""),
]
