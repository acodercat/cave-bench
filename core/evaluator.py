"""Benchmark evaluation engine for cave-bench.

This module provides the Evaluator class which can evaluate
any agent implementation that conforms to the Agent interface.
"""

import logging
import ast
import copy
from typing import Dict, List, Callable, Set, Optional
from core.validation import validate_function_calls, ErrorType, ValidationError, ValidatorResult
from core.types import (
    ToolCall, TurnMetrics, TurnResult,
    ConversationResult, ScenarioMetrics, ScenarioResult, VariableAccess,
    Turn, Conversation, ExpectedFunctionCall, BenchmarkScenario
)
from core.agent import Agent, AgentFactory, AgentToolCall


logger = logging.getLogger('Agent.Evaluator')


def analyze_variable_access(code: str) -> VariableAccess:
    """
    Analyze the variable access in a Python code string.

    Args:
        code: Python code string

    Returns:
        VariableAccess with reads and writes lists
    """
    try:
        tree = ast.parse(code)
        analyzer = VariableAnalyzer()
        analyzer.visit(tree)

        return VariableAccess(
            reads=list(analyzer.reads),
            writes=list(analyzer.writes)
        )
    except SyntaxError:
        # If AST parsing fails, return empty results
        return VariableAccess(reads=[], writes=[])


class VariableAnalyzer(ast.NodeVisitor):
    """A simple variable access analyzer"""

    def __init__(self):
        self.reads: Set[str] = set()
        self.writes: Set[str] = set()

    def visit_Name(self, node):
        """Visit variable names"""
        if isinstance(node.ctx, ast.Load):
            # Variable read
            self.reads.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            # Variable write
            self.writes.add(node.id)

        self.generic_visit(node)


class Evaluator:
    """Evaluator for benchmarking agent implementations.

    This evaluator works with any agent that implements the Agent interface,
    including CaveAgent (Python code execution) and LiteLLM (JSON function calling).
    """

    def __init__(self, agent_factory: AgentFactory):
        """
        Initialize the benchmark evaluator.

        Args:
            agent_factory: Factory for creating agent instances
        """
        self.agent_factory = agent_factory
        self._reset_metrics()

    def _reset_metrics(self):
        """
        Reset all evaluation metrics to their initial values.
        This should be called before starting a new evaluation.
        """
        logger.debug("Resetting evaluation metrics")
        self.metrics = ScenarioMetrics()

    async def evaluate(
        self,
        scenario: str,
        module,
        conversations: List[Conversation],
        json_config: dict | None = None
    ) -> ScenarioResult:
        """
        Evaluate an agent against a specific scenario.

        Args:
            scenario: The scenario name
            module: The scenario module
            conversations: List of benchmark conversations to evaluate
            json_config: Optional JSON config dict containing 'description' and 'requirements'

        Returns:
            ScenarioResult with complete evaluation results
        """
        # Reset metrics at the start of evaluation
        self._reset_metrics()

        # Load scenario contents with optional JSON overrides for prompts
        scenario_contents = BenchmarkScenario.from_module(module, json_config)

        logger.info(f"Evaluating scenario {scenario} with {len(conversations)} conversations")

        # Process each conversation
        conversation_results = []
        for conversation in conversations:
            result = await self._evaluate_conversation(
                conversation, scenario_contents
            )
            conversation_results.append(result)

        # Calculate success rate
        self.metrics.success_rate = (
            self.metrics.successful_turns / self.metrics.total_turns
            if self.metrics.total_turns > 0 else 0
        )

        logger.info(f"Evaluation complete. Success rate: {self.metrics.success_rate:.2f}")

        return ScenarioResult(
            scenario=scenario,
            conversations=conversation_results,
            metrics=self.metrics
        )

    async def _evaluate_conversation(
        self,
        conversation: Conversation,
        scenario: BenchmarkScenario
    ) -> ConversationResult:
        """Evaluate a single conversation within a scenario."""
        logger.debug(f"Evaluating conversation: {conversation.id}")

        # Deep copy variables to ensure fresh state for each conversation
        # This prevents mutable values (lists, dicts) from persisting across conversations
        fresh_variables = copy.deepcopy(scenario.variables)

        # Create agent using the factory
        agent = self.agent_factory.create_agent(
            functions=scenario.tools,
            variables=fresh_variables,
            types=scenario.types,
            description=scenario.description,
            requirements=scenario.requirements
        )

        # Process each turn
        turn_results = []
        for turn in conversation.turns:
            result = await self._evaluate_turn(
                turn, agent, scenario.validators, scenario.hooks
            )
            turn_results.append(result)
            self.metrics.total_steps += result.metrics.steps

        return ConversationResult(
            id=conversation.id,
            turns=turn_results
        )

    def _calculate_missing_calls_metric(
        self,
        expected_calls: List[ExpectedFunctionCall],
        actual_calls: List[AgentToolCall]
    ) -> int:
        """
        Calculate the number of missing required function calls.

        Args:
            expected_calls: List of expected function calls from benchmarks
            actual_calls: List of actual function calls from the agent

        Returns:
            Number of missing required function calls
        """
        missing_required_calls = 0

        # Create a set of function names from actual calls for quick lookup
        actual_function_names = {call.function for call in actual_calls}

        # Count missing required calls
        for expected in expected_calls:
            if expected.required and expected.name not in actual_function_names:
                missing_required_calls += 1

        return missing_required_calls

    async def _evaluate_turn(
        self,
        turn: Turn,
        agent: Agent,
        validators: Optional[Dict[str, Callable]] = None,
        hooks: Optional[Dict[str, Callable]] = None
    ) -> TurnResult:
        """Evaluate a single turn within a conversation and return detailed metrics."""
        if validators is None:
            validators = {}
        if hooks is None:
            hooks = {}

        query = turn.query
        reference_response = turn.reference_response
        expected_calls = turn.expected_function_calls
        expected_variable_reads = turn.expected_variable_reads
        expected_variable_writes = turn.expected_variable_writes

        # Call pre-turn hook if specified (injects randomness / modifies state)
        # Note: hooks require runtime access, only available for CaveAgent
        if turn.pre_turn_hook:
            if turn.pre_turn_hook not in hooks:
                raise KeyError(f"Hook '{turn.pre_turn_hook}' not found. Available hooks: {list(hooks.keys())}")
            if agent.runtime is None:
                raise ValueError(f"Hook '{turn.pre_turn_hook}' requires runtime access, but agent has no runtime")
            hook_result = hooks[turn.pre_turn_hook](agent.runtime, turn)
            # Hook can return a new query string, or None to use the original
            if hook_result is not None:
                query = hook_result
                logger.debug(f"Pre-turn hook modified query to: {query[:100]}...")

        # Initialize metrics
        turn_metrics = TurnMetrics()

        self.metrics.total_expected_calls += len(expected_calls)
        self.metrics.total_turns += 1

        # Run the agent
        result = await agent.run(query)

        turn_metrics.steps = result.steps

        # Track token usage
        token_usage = result.token_usage
        turn_metrics.prompt_tokens = token_usage.prompt_tokens
        turn_metrics.completion_tokens = token_usage.completion_tokens
        turn_metrics.total_tokens = token_usage.total_tokens
        self.metrics.total_prompt_tokens += token_usage.prompt_tokens
        self.metrics.total_completion_tokens += token_usage.completion_tokens
        self.metrics.total_tokens += token_usage.total_tokens

        # Analyze code snippets for variable access (only for agents that produce code)
        code_snippets = result.code_snippets
        actual_variable_reads = []
        actual_variable_writes = []

        for code_snippet in code_snippets:
            variable_access = analyze_variable_access(code_snippet)
            for read in variable_access.reads:
                if read in expected_variable_reads:
                    actual_variable_reads.append(read)
            for write in variable_access.writes:
                if write in expected_variable_writes:
                    actual_variable_writes.append(write)

        # Get function calls from agent response
        actual_calls = result.tool_calls
        actual_calls_dict = [call.to_dict() for call in actual_calls]
        self.metrics.total_actual_calls += len(actual_calls)

        # Convert AgentToolCall to ToolCall for validation
        tool_calls_for_validation = [
            ToolCall(
                function=call.function,
                arguments=call.arguments,
                call_id=call.call_id
            )
            for call in actual_calls
        ]

        # Run validator if specified (after we have actual_calls)
        # Note: validators may require runtime access
        validator_name = turn.validator
        if validator_name:
            if validator_name not in validators:
                raise KeyError(f"Validator '{validator_name}' not found. Available validators: {list(validators.keys())}")
            validator_result = validators[validator_name](
                result.content, agent.runtime, turn, tool_calls_for_validation
            )
        else:
            validator_result = ValidatorResult(success=True, message="")

        # Validate function calls
        validation_errors = validate_function_calls(tool_calls_for_validation, expected_calls)

        # Check for missing variable reads/writes
        for expected_variable_read in expected_variable_reads:
            if expected_variable_read not in actual_variable_reads:
                validation_errors.append(ValidationError(
                    error_type=ErrorType.MISSING_VARIABLE_READ,
                    message=f"Variable {expected_variable_read} is not read",
                ))
        for expected_variable_write in expected_variable_writes:
            if expected_variable_write not in actual_variable_writes:
                validation_errors.append(ValidationError(
                    error_type=ErrorType.MISSING_VARIABLE_WRITE,
                    message=f"Variable {expected_variable_write} is not written",
                ))

        # Store error messages for display
        error_messages = [error.message for error in validation_errors]
        if not validator_result.success:
            error_messages.append(f"Custom validator failed: {validator_result.message}")

        # Calculate missing calls metric
        turn_metrics.missing_calls = self._calculate_missing_calls_metric(expected_calls, actual_calls)
        self.metrics.missing_calls += turn_metrics.missing_calls

        # Count error types
        for error in validation_errors:
            if error.error_type == ErrorType.WRONG_ARGUMENT_TYPE:
                self.metrics.wrong_argument_types += 1
                turn_metrics.wrong_argument_types += 1
            elif error.error_type == ErrorType.WRONG_ARGUMENT_VALUE:
                self.metrics.wrong_argument_values += 1
                turn_metrics.wrong_argument_values += 1
            elif error.error_type == ErrorType.MISSING_ARGUMENT:
                self.metrics.missing_arguments += 1
                turn_metrics.missing_arguments += 1
            elif error.error_type == ErrorType.MISSING_VARIABLE_READ:
                self.metrics.missing_variable_reads += 1
                turn_metrics.missing_variable_reads += 1
            elif error.error_type == ErrorType.MISSING_VARIABLE_WRITE:
                self.metrics.missing_variable_writes += 1
                turn_metrics.missing_variable_writes += 1

        # Determine turn success (ensure native Python bool for JSON serialization)
        success = bool(not validation_errors and validator_result.success)
        if success:
            self.metrics.successful_turns += 1
        else:
            self.metrics.failed_turns += 1

        return TurnResult(
            query=query,
            reference_response=reference_response,
            actual_response=result.content,
            expected_calls=[call.to_dict() for call in expected_calls],
            actual_calls=actual_calls_dict,
            validation_errors=error_messages,
            metrics=turn_metrics,
            success=success,
            validator_result=bool(validator_result.success),
            code_snippets=code_snippets,
            expected_variable_reads=expected_variable_reads,
            expected_variable_writes=expected_variable_writes,
            actual_variable_reads=actual_variable_reads,
            actual_variable_writes=actual_variable_writes
        )
