"""Function call tracker using Python's profiling hooks."""

import threading
import inspect
from types import FrameType
from typing import Any, List, Optional, Type
from core.types import ToolCall


class FunctionCallTracker:
    """Tracker for function calls using Python's threading.setprofile (thread-safe)."""

    def __init__(self, target_functions: Optional[List[str]] = None) -> None:
        """
        Initialize a function call tracker.

        Args:
            target_functions: List of function names to track. If None, track all functions.
        """
        self.tool_calls: List[ToolCall] = []
        self.current_call_id: int = 0
        self.target_functions = target_functions

    def start(self) -> None:
        """Start tracking function calls and returns"""
        self.tool_calls = []
        self.current_call_id = 0
        threading.setprofile(self._profile_handler)

    def stop(self) -> None:
        """Stop tracking function calls and returns"""
        threading.setprofile(None)

    def __enter__(self) -> "FunctionCallTracker":
        """Context manager entry - start tracking."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        """Context manager exit - ensure tracking is stopped even on exception."""
        self.stop()

    def _profile_handler(self, frame: FrameType, event: str, arg: Any) -> None:
        """Profile handler to capture function calls and returns"""
        if event != 'call':
            return

        # Get function name
        func_name = frame.f_code.co_name

        # If target_functions is specified, only track those functions
        if self.target_functions is not None and func_name not in self.target_functions:
            return

        # Increment call ID for this function call
        self.current_call_id += 1
        call_id = self.current_call_id

        # Get function arguments
        arg_info = inspect.getargvalues(frame)
        arguments = {}

        # Process regular arguments
        for arg_name in arg_info.args:
            if arg_name == 'self':  # Skip self in methods
                continue

            if arg_name in arg_info.locals:
                value = arg_info.locals[arg_name]
                arguments[arg_name] = value

        tool_call = ToolCall(
            call_id=str(call_id),
            function=func_name,
            arguments=arguments
        )

        # Add to call history
        self.tool_calls.append(tool_call)

    def get_tool_calls(self) -> List[ToolCall]:
        """Get tracked tool calls"""
        return self.tool_calls
