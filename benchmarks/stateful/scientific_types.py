"""
Scientific Types - Stateful Benchmark

Tests: pandas DataFrame, numpy ndarray, pandas Series.
"""

from typing import List
import pandas as pd
import numpy as np
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# INITIAL STATE
# ============================================================================

# Main DataFrame for various operations
df = pd.DataFrame({
    "category": ["Electronics", "Electronics", "Clothing", "Clothing", "Food"],
    "product": ["Phone", "Laptop", "Shirt", "Pants", "Apple"],
    "price": [500.0, 1200.0, 50.0, 80.0, 10.0],
    "quantity": [10, 5, 20, 15, 100]
})

# Second DataFrame for merge operations
df2 = pd.DataFrame({
    "product": ["Phone", "Laptop", "Shirt"],
    "supplier": ["SupA", "SupB", "SupA"]
})

# Main array for matrix operations
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Second array for operations
array = np.array([10, 25, 5, 30, 15, 8, 22, 3])

# Main Series
series = pd.Series([85, 92, 78, 95, 88], index=["a", "b", "c", "d", "e"], name="scores")

# Result variables (reused across conversations)
result_df = None
result_array = None
result_value = 0.0


# ============================================================================
# VALIDATORS - DataFrame Operations
# ============================================================================

def validate_df_add_column(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    d = runtime.get_variable("df")
    if "total" not in d.columns:
        return ValidatorResult(False, "Column 'total' not found")
    expected = [5000.0, 6000.0, 1000.0, 1200.0, 1000.0]
    actual = d["total"].tolist()
    ok = actual == expected
    return ValidatorResult(ok, f"total = {actual}")

def validate_df_filter(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    d = runtime.get_variable("df")
    ok = len(d) == 3 and all(t > 1000 for t in d["total"].tolist())
    return ValidatorResult(ok, f"df has {len(d)} rows with total > 1000")

def validate_df_sort(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    d = runtime.get_variable("df")
    totals = d["total"].tolist()
    ok = totals == sorted(totals, reverse=True)
    return ValidatorResult(ok, f"totals = {totals}")


# ============================================================================
# VALIDATORS - DataFrame Groupby
# ============================================================================

def validate_df_groupby(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_df")
    if r is None:
        return ValidatorResult(False, "result_df is None")
    # Electronics: 1700, Clothing: 130, Food: 10
    if isinstance(r, pd.Series):
        r = r.to_dict()
    elif isinstance(r, pd.DataFrame):
        r = r.set_index("category")["price"].to_dict() if "category" in r.columns else r.to_dict()
    expected = {"Electronics": 1700.0, "Clothing": 130.0, "Food": 10.0}
    ok = r == expected or (isinstance(r, dict) and all(abs(r.get(k, 0) - v) < 0.01 for k, v in expected.items()))
    return ValidatorResult(ok, f"grouped = {r}")

def validate_df_groupby_count(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_value")
    # Number of categories: 3
    ok = r == 3
    return ValidatorResult(ok, f"category count = {r}, expected 3")

def validate_df_groupby_max(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_value")
    # Max total price: Electronics = 1700
    ok = abs(r - 1700.0) < 0.01
    return ValidatorResult(ok, f"max = {r}, expected 1700.0")


# ============================================================================
# VALIDATORS - DataFrame Merge
# ============================================================================

def validate_df_merge(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_df")
    if r is None:
        return ValidatorResult(False, "result_df is None")
    ok = len(r) == 3 and "supplier" in r.columns
    return ValidatorResult(ok, f"merged has {len(r)} rows, columns: {r.columns.tolist()}")

def validate_df_merge_filter(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_df")
    if r is None:
        return ValidatorResult(False, "result_df is None")
    # Filter supplier == 'SupA': Phone, Shirt
    ok = len(r) == 2 and set(r["product"].tolist()) == {"Phone", "Shirt"}
    return ValidatorResult(ok, f"filtered to {r['product'].tolist()}")

def validate_df_merge_sum(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_value")
    # Sum of prices for SupA: 500 + 50 = 550
    ok = abs(r - 550.0) < 0.01
    return ValidatorResult(ok, f"sum = {r}, expected 550.0")


# ============================================================================
# VALIDATORS - ndarray Operations
# ============================================================================

def validate_array_multiply(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    m = runtime.get_variable("matrix")
    expected = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
    ok = np.array_equal(m, expected)
    return ValidatorResult(ok, f"matrix[0] = {m[0].tolist()}")

def validate_array_add(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    m = runtime.get_variable("matrix")
    expected = np.array([[12, 14, 16], [18, 20, 22], [24, 26, 28]])
    ok = np.array_equal(m, expected)
    return ValidatorResult(ok, f"matrix[0] = {m[0].tolist()}")

def validate_array_transpose(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    m = runtime.get_variable("matrix")
    expected = np.array([[12, 18, 24], [14, 20, 26], [16, 22, 28]])
    ok = np.array_equal(m, expected)
    return ValidatorResult(ok, f"matrix shape = {m.shape}")


# ============================================================================
# VALIDATORS - ndarray Reshape
# ============================================================================

def validate_array_reshape(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_array")
    if r is None:
        return ValidatorResult(False, "result_array is None")
    ok = r.shape == (2, 4)
    return ValidatorResult(ok, f"shape = {r.shape}, expected (2, 4)")

def validate_array_sum_axis(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_array")
    if r is None:
        return ValidatorResult(False, "result_array is None")
    # Sum along axis 1: [10+25+5+30, 15+8+22+3] = [70, 48]
    expected = np.array([70, 48])
    ok = np.array_equal(r, expected)
    return ValidatorResult(ok, f"row sums = {r.tolist()}")

def validate_array_total(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_value")
    # Total: 70 + 48 = 118
    ok = r == 118
    return ValidatorResult(ok, f"total = {r}, expected 118")


# ============================================================================
# VALIDATORS - ndarray Boolean
# ============================================================================

def validate_array_bool_filter(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_array")
    if r is None:
        return ValidatorResult(False, "result_array is None")
    expected = np.array([25, 30, 15, 22])
    ok = np.array_equal(r, expected)
    return ValidatorResult(ok, f"filtered = {r.tolist()}")

def validate_array_bool_replace(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    a = runtime.get_variable("array")
    expected = np.array([0, 25, 0, 30, 15, 0, 22, 0])
    ok = np.array_equal(a, expected)
    return ValidatorResult(ok, f"array = {a.tolist()}")

def validate_array_bool_mean(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_value")
    # Mean of non-zero: (25+30+15+22)/4 = 23.0
    ok = abs(r - 23.0) < 0.01
    return ValidatorResult(ok, f"mean = {r}, expected 23.0")


# ============================================================================
# VALIDATORS - Series Operations
# ============================================================================

def validate_series_add(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    s = runtime.get_variable("series")
    expected = [90, 97, 83, 100, 93]
    ok = s.tolist() == expected
    return ValidatorResult(ok, f"series = {s.tolist()}")

def validate_series_clip(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    s = runtime.get_variable("series")
    ok = s.max() <= 100 and s.min() >= 80
    return ValidatorResult(ok, f"min={s.min()}, max={s.max()}")

def validate_series_mean(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_value")
    s = runtime.get_variable("series")
    expected = s.mean()
    ok = abs(r - expected) < 0.01
    return ValidatorResult(ok, f"mean = {r}")


# ============================================================================
# VALIDATORS - Series Index
# ============================================================================

def validate_series_select(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_array")
    if r is None:
        return ValidatorResult(False, "result_array is None")
    # Select b, c, d from original: 92, 78, 95
    if hasattr(r, 'tolist'):
        r = r.tolist()
    ok = list(r) == [92, 78, 95]
    return ValidatorResult(ok, f"selected = {list(r)}")

def validate_series_multiply(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    s = runtime.get_variable("series")
    # b, c, d doubled: a=85, b=184, c=156, d=190, e=88
    expected = {"a": 85, "b": 184, "c": 156, "d": 190, "e": 88}
    ok = s.to_dict() == expected
    return ValidatorResult(ok, f"series = {s.to_dict()}")

def validate_series_sum(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_value")
    # Sum: 85 + 184 + 156 + 190 + 88 = 703
    ok = r == 703
    return ValidatorResult(ok, f"sum = {r}, expected 703")


# ============================================================================
# VALIDATORS - Series Boolean
# ============================================================================

def validate_series_bool_filter(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_array")
    if r is None:
        return ValidatorResult(False, "result_array is None")
    # Values > 90 from original: 92, 95
    if hasattr(r, 'tolist'):
        r = r.tolist()
    ok = sorted(r) == [92, 95]
    return ValidatorResult(ok, f"filtered = {list(r)}")

def validate_series_bool_count(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_value")
    # Count > 90: 2
    ok = r == 2
    return ValidatorResult(ok, f"count = {r}, expected 2")

def validate_series_max(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_value")
    ok = r == 95
    return ValidatorResult(ok, f"max = {r}, expected 95")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    # Data variables
    Variable("df", df, "DataFrame with columns: category, product, price, quantity."),
    Variable("df2", df2, "DataFrame with columns: product, supplier."),
    Variable("matrix", matrix, "3x3 numpy array: [[1,2,3],[4,5,6],[7,8,9]]."),
    Variable("array", array, "1D numpy array: [10, 25, 5, 30, 15, 8, 22, 3]."),
    Variable("series", series, "Series with index a-e, values [85, 92, 78, 95, 88]."),
    # Result variables
    Variable("result_df", result_df, "Store DataFrame results here."),
    Variable("result_array", result_array, "Store array/Series results here."),
    Variable("result_value", result_value, "Store numeric results here (float)."),
]

validators = {
    # DataFrame ops
    "validate_df_add_column": validate_df_add_column,
    "validate_df_filter": validate_df_filter,
    "validate_df_sort": validate_df_sort,
    # DataFrame groupby
    "validate_df_groupby": validate_df_groupby,
    "validate_df_groupby_count": validate_df_groupby_count,
    "validate_df_groupby_max": validate_df_groupby_max,
    # DataFrame merge
    "validate_df_merge": validate_df_merge,
    "validate_df_merge_filter": validate_df_merge_filter,
    "validate_df_merge_sum": validate_df_merge_sum,
    # ndarray ops
    "validate_array_multiply": validate_array_multiply,
    "validate_array_add": validate_array_add,
    "validate_array_transpose": validate_array_transpose,
    # ndarray reshape
    "validate_array_reshape": validate_array_reshape,
    "validate_array_sum_axis": validate_array_sum_axis,
    "validate_array_total": validate_array_total,
    # ndarray boolean
    "validate_array_bool_filter": validate_array_bool_filter,
    "validate_array_bool_replace": validate_array_bool_replace,
    "validate_array_bool_mean": validate_array_bool_mean,
    # Series ops
    "validate_series_add": validate_series_add,
    "validate_series_clip": validate_series_clip,
    "validate_series_mean": validate_series_mean,
    # Series index
    "validate_series_select": validate_series_select,
    "validate_series_multiply": validate_series_multiply,
    "validate_series_sum": validate_series_sum,
    # Series boolean
    "validate_series_bool_filter": validate_series_bool_filter,
    "validate_series_bool_count": validate_series_bool_count,
    "validate_series_max": validate_series_max,
}
