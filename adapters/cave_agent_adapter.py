"""CaveAgent adapter for cave-bench benchmarking framework.

This adapter wraps CaveAgent to conform to the abstract Agent interface,
allowing it to be evaluated using the same pipeline as JSON function calling agents.
"""

from typing import List, Callable, Optional
from core.agent import Agent, AgentFactory, AgentResponse, TokenUsage
from core.tracker import FunctionCallTracker
from core.prompts import DEFAULT_AGENT_IDENTITY, DEFAULT_INSTRUCTIONS
from cave_agent import CaveAgent, LogLevel, Model
from cave_agent.runtime import PythonRuntime, Function, Variable, Type


class CaveAgentWrapper(Agent):
    """Agent implementation that wraps CaveAgent.

    CaveAgent executes Python code to call functions, so we use
    sys.setprofile() via FunctionCallTracker to capture actual function calls.
    """

    def __init__(
        self,
        model: Model,
        functions: List[Callable],
        variables: Optional[List[Variable]] = None,
        types: Optional[List[Type]] = None,
        description: Optional[str] = None,
        requirements: Optional[str] = None
    ):
        """Initialize the CaveAgentWrapper.

        Args:
            model: The LLM model configuration
            functions: List of callable functions/tools
            variables: List of variables for stateful execution
            types: List of custom types
            description: Scenario-specific agent description
            requirements: Scenario-specific requirements
        """
        self._model = model
        self._functions = functions
        self._function_names = [f.__name__ for f in functions]
        self._variables = variables or []
        self._types = types or []

        # Build instructions
        instructions = DEFAULT_INSTRUCTIONS
        if description:
            instructions = instructions + "\nTASK DESCRIPTION: \n" + description + "\n"
        if requirements:
            instructions = instructions + "\nTASK REQUIREMENTS: \n" + requirements

        # Create runtime with wrapped functions
        wrapped_functions = [Function(f) for f in functions]
        runtime = PythonRuntime(
            functions=wrapped_functions,
            variables=self._variables,
            types=self._types
        )

        # Create the underlying CaveAgent
        self._agent = CaveAgent(
            model=model,
            runtime=runtime,
            max_steps=100,
            max_history=200,
            max_exec_output=80000,
            log_level=LogLevel.DEBUG,
        )

    @property
    def runtime(self) -> PythonRuntime:
        """Get the Python runtime for accessing variables."""
        return self._agent.runtime

    async def run(self, query: str) -> AgentResponse:
        """Run the agent and capture function calls via profiling.

        Args:
            query: The user input query

        Returns:
            AgentResponse with result, tool calls, steps, code snippets, and token usage
        """
        # Use context manager to safely track function calls
        with FunctionCallTracker(target_functions=self._function_names) as tracker:
            result = await self._agent.run(query)

        # Get tracked tool calls
        tool_calls = tracker.get_tool_calls()

        # Extract token usage from CaveAgent's response
        cave_token_usage = result.token_usage
        token_usage = TokenUsage(
            prompt_tokens=cave_token_usage.prompt_tokens,
            completion_tokens=cave_token_usage.completion_tokens,
            total_tokens=cave_token_usage.total_tokens
        )

        return AgentResponse(
            content=result.content,
            tool_calls=tool_calls,
            steps=result.steps_taken,
            code_snippets=result.code_snippets,
            token_usage=token_usage
        )


class CaveAgentFactory(AgentFactory):
    """Factory for creating CaveAgent instances."""

    def __init__(self, model: Model):
        """Initialize the factory with a model configuration.

        Args:
            model: The LLM model to use for all created agents
        """
        self.model = model

    def create_agent(
        self,
        functions: List[Callable],
        variables: Optional[List[Variable]] = None,
        types: Optional[List[Type]] = None,
        description: Optional[str] = None,
        requirements: Optional[str] = None
    ) -> CaveAgentWrapper:
        """Create a CaveAgent with the specified configuration.

        Args:
            functions: List of callable functions/tools
            variables: List of variables for stateful execution
            types: List of custom types
            description: Scenario-specific agent description
            requirements: Scenario-specific requirements

        Returns:
            A CaveAgentWrapper instance
        """
        return CaveAgentWrapper(
            model=self.model,
            functions=functions,
            variables=variables,
            types=types,
            description=description,
            requirements=requirements
        )
