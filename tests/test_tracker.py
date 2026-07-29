"""Tests for FunctionCallTracker (sys.setprofile-based tool-call capture).

The load-bearing behaviour: only the agent-initiated (outermost) tool call is
recorded. When a tool's body calls another tracked tool, that inner call must
NOT be counted as a separate agent call.
"""

from core.tracker import FunctionCallTracker


def _inner(x):
    return x + 1


def _outer(x):
    # Calls another tracked tool internally — must not be double-counted.
    return _inner(x) * 2


def _other(y):
    return y


class TestFunctionCallTracker:
    def test_records_top_level_call_only(self):
        with FunctionCallTracker(target_functions=["_outer", "_inner"]) as t:
            _outer(1)
        calls = t.get_tool_calls()
        # _inner is called inside _outer, so it is NOT an agent call.
        assert [c.function for c in calls] == ["_outer"]
        assert calls[0].arguments == {"x": 1}

    def test_two_separate_top_level_calls(self):
        with FunctionCallTracker(target_functions=["_outer", "_other"]) as t:
            _outer(1)
            _other(2)
        assert [c.function for c in t.get_tool_calls()] == ["_outer", "_other"]

    def test_recursion_records_outermost_only(self):
        def recurse(n):
            return recurse(n - 1) if n > 0 else 0

        with FunctionCallTracker(target_functions=["recurse"]) as t:
            recurse(3)
        assert len(t.get_tool_calls()) == 1

    def test_non_target_calls_ignored(self):
        with FunctionCallTracker(target_functions=["_outer"]) as t:
            _other(5)      # not a tracked tool
            _outer(1)
        assert [c.function for c in t.get_tool_calls()] == ["_outer"]

    def test_arguments_captured(self):
        def tool(a, b=2):
            return a + b

        with FunctionCallTracker(target_functions=["tool"]) as t:
            tool(10, b=5)
        assert t.get_tool_calls()[0].arguments == {"a": 10, "b": 5}

    def test_track_all_mode_keeps_nested(self):
        # target_functions=None = "track everything"; no nesting skip applies.
        with FunctionCallTracker(target_functions=None) as t:
            _outer(1)
        names = [c.function for c in t.get_tool_calls()]
        assert "_outer" in names and "_inner" in names
