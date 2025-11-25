"""Core module for cave-bench benchmarking framework."""

from core.types import (
    ToolCall,
    VariableAccess,
    ExpectedArgument,
    ExpectedFunctionCall,
    BenchmarkTurn,
    BenchmarkConversation,
    BenchmarkScenario,
    TurnMetrics,
    TurnResult,
    ConversationResult,
    ScenarioMetrics,
    ScenarioResult,
)
from core.tracker import FunctionCallTracker
from core.evaluator import BenchmarkEvaluator, analyze_variable_access
from core.validation import (
    ErrorType,
    ValidationError,
    ValidatorResult,
    validate_function_calls,
    validate_arguments,
    normalize_value,
    is_type_compatible,
)
from core.prompts import DEFAULT_AGENT_IDENTITY, DEFAULT_INSTRUCTIONS

__all__ = [
    # Types
    "ToolCall",
    "VariableAccess",
    "ExpectedArgument",
    "ExpectedFunctionCall",
    "BenchmarkTurn",
    "BenchmarkConversation",
    "BenchmarkScenario",
    "TurnMetrics",
    "TurnResult",
    "ConversationResult",
    "ScenarioMetrics",
    "ScenarioResult",
    # Tracker
    "FunctionCallTracker",
    # Evaluator
    "BenchmarkEvaluator",
    "analyze_variable_access",
    # Validation
    "ErrorType",
    "ValidationError",
    "ValidatorResult",
    "validate_function_calls",
    "validate_arguments",
    "normalize_value",
    "is_type_compatible",
    # Prompts (defaults)
    "DEFAULT_AGENT_IDENTITY",
    "DEFAULT_INSTRUCTIONS",
]
