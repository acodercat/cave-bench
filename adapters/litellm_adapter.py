"""LiteLLM adapter for JSON function calling evaluation.

This adapter wraps LiteLLM to evaluate traditional JSON-based function calling,
allowing comparison with CaveAgent's Python code execution approach.

The agent implements a proper agentic loop:
1. Send query to model
2. If model returns tool calls, execute them and feed results back
3. Repeat until model returns final response
"""

from typing import List, Callable, Optional, Any, Dict
import json
import logging
from core.agent import Agent, AgentFactory, AgentResponse, TokenUsage
from core.types import ToolCall
from litellm import acompletion

logger = logging.getLogger('Agent.LitellmAdapter')

# Default system prompt for JSON function calling
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools/functions.
When the user asks you to perform a task, analyze what tools you need and call them appropriately.
Always use the tools when they can help accomplish the user's request.
After getting tool results, provide a helpful response to the user."""


def function_to_schema(func: Callable) -> Dict[str, Any]:
    """
    Convert a Python function to an OpenAI-compatible tool schema.

    Args:
        func: Python function with type hints and docstring

    Returns:
        Dictionary with name, description, and parameters schema

    Note:
        This requires the 'agents' package. Used for compatibility
        with non-CaveAgent frameworks.
    """
    try:
        from agents import function_tool
    except ImportError:
        raise ImportError(
            "The 'agents' package is required for function_to_schema. "
            "Install it with: pip install openai-agents"
        )

    tool = function_tool(func, strict_mode=False)
    return {
        "name": func.__name__,
        "description": func.__doc__ or tool.description,
        "parameters": tool.params_json_schema
    }

class LitellmModel:
    """Model configuration for LiteLLM.

    Supports multiple providers through LiteLLM's unified API:
    - OpenAI (provider='openai')
    - DeepSeek (provider='deepseek')
    - Google Gemini (provider='gemini')
    - Anthropic Claude (provider='anthropic')
    - And many more via LiteLLM
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        provider: str,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None
    ):
        """Initialize the LiteLLM model configuration.

        Args:
            model_id: The model identifier (e.g., 'gpt-4', 'deepseek-chat')
            api_key: API key for the provider
            provider: LiteLLM provider name (e.g., 'openai', 'deepseek')
            temperature: Optional temperature setting
            base_url: Optional custom API base URL
        """
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        self.temperature = temperature


class LitellmAgentWrapper(Agent):
    """Agent that uses LiteLLM for JSON-based function calling.

    Implements a proper agentic loop:
    - Maintains conversation history across turns
    - Executes tool calls and feeds results back to the model
    - Loops until model returns final response (no tool calls)
    - Uses system prompt built from description and requirements
    """

    def __init__(
        self,
        model: LitellmModel,
        tools: List[Callable],
        max_steps: int = 20,
        description: Optional[str] = None,
        requirements: Optional[str] = None
    ):
        """Initialize the LiteLLM agent wrapper.

        Args:
            model: The LiteLLM model configuration
            tools: List of callable functions to expose to the model
            max_steps: Maximum number of steps (API calls) per run
            description: Task description for system prompt
            requirements: Task requirements for system prompt
        """
        self._model = model
        self._max_steps = max_steps
        self._total_steps = 0
        self._total_token_usage = TokenUsage()

        # Build system prompt from description and requirements
        self._system_prompt = self._build_system_prompt(description, requirements)

        # Initialize messages with system prompt
        self._messages: List[Dict[str, Any]] = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})

        # Store tools as dict for execution lookup
        self._tools_dict: Dict[str, Callable] = {
            tool.__name__: tool for tool in tools
        }

        # Convert to OpenAI tool format for API calls
        self._tools_schema = [
            {"type": "function", "function": function_to_schema(tool)}
            for tool in tools
        ]

    def _build_system_prompt(
        self,
        description: Optional[str],
        requirements: Optional[str]
    ) -> str:
        """Build system prompt from description and requirements.

        Args:
            description: Task description
            requirements: Task requirements

        Returns:
            Formatted system prompt string
        """
        parts = [DEFAULT_SYSTEM_PROMPT]

        if description:
            parts.append(f"\n\nTASK DESCRIPTION:\n{description}")

        if requirements:
            parts.append(f"\n\nTASK REQUIREMENTS:\n{requirements}")

        return "".join(parts)

    def _execute_tool(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result as a string.

        Args:
            function_name: Name of the function to call
            arguments: Arguments to pass to the function

        Returns:
            String representation of the result
        """
        if function_name not in self._tools_dict:
            return f"Error: Unknown function '{function_name}'"

        try:
            func = self._tools_dict[function_name]
            result = func(**arguments)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            logger.error(f"Tool execution failed: {function_name}({arguments}): {e}")
            return f"Error: {str(e)}"

    async def _call_model(self) -> Any:
        """Make an API call to the model.

        Returns:
            The LiteLLM response object
        """
        extra_params = {}
        if "gemini" in self._model.model_id.lower():
            extra_params["reasoning_effort"] = "low"

        return await acompletion(
            model=self._model.model_id,
            api_key=self._model.api_key,
            base_url=self._model.base_url,
            custom_llm_provider=self._model.provider,
            tools=self._tools_schema if self._tools_schema else None,
            parallel_tool_calls=True,
            tool_choice="auto" if self._tools_schema else None,
            messages=self._messages,
            temperature=self._model.temperature,
            **extra_params
        )

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from LiteLLM response.

        Args:
            response: The LiteLLM response object

        Returns:
            TokenUsage with extracted values
        """
        if hasattr(response, 'usage') and response.usage:
            return TokenUsage(
                prompt_tokens=getattr(response.usage, 'prompt_tokens', 0) or 0,
                completion_tokens=getattr(response.usage, 'completion_tokens', 0) or 0,
                total_tokens=getattr(response.usage, 'total_tokens', 0) or 0
            )
        return TokenUsage()

    async def run(self, query: str) -> AgentResponse:
        """Run the agent with a query using the agentic loop.

        The agent will:
        1. Add user query to message history
        2. Call model
        3. If model returns tool calls, execute them and add results to history
        4. Repeat until model returns final response or max_steps reached

        Args:
            query: The user input query

        Returns:
            AgentResponse with result, all tool calls made, step count, and token usage
        """
        # Add user message to history
        self._messages.append({"role": "user", "content": query})

        all_tool_calls: List[ToolCall] = []
        steps = 0
        total_token_usage = TokenUsage()

        while steps < self._max_steps:
            steps += 1

            try:
                response = await self._call_model()
            except Exception as e:
                logger.error(f"LiteLLM API call failed: {e}")
                return AgentResponse(
                    content=f"Error: {str(e)}",
                    tool_calls=all_tool_calls,
                    steps=steps,
                    token_usage=total_token_usage
                )

            # Accumulate token usage from this API call
            total_token_usage = total_token_usage + self._extract_token_usage(response)

            choice = response.choices[0]
            assistant_message = choice.message

            # Check if model wants to call tools
            if choice.finish_reason == "tool_calls" and assistant_message.tool_calls:

                logger.debug(f"Tool calls: {assistant_message.tool_calls}")
                logger.debug(f"Assistant message: {assistant_message}")
                
                # Add assistant message with tool calls to history
                self._messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })

                # Execute each tool call and add results
                for tool_call in assistant_message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse arguments: {tool_call.function.arguments}")
                        arguments = {}

                    # Record the tool call
                    agent_tool_call = ToolCall(
                        call_id=tool_call.id,
                        function=tool_call.function.name,
                        arguments=arguments
                    )
                    all_tool_calls.append(agent_tool_call)

                    # Execute the tool
                    result = self._execute_tool(tool_call.function.name, arguments)

                    # Add tool result to message history
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                # Continue the loop to get model's next response
                continue

            else:
                # Model returned final response (no tool calls)
                content = assistant_message.content or ""

                # Add final assistant message to history
                self._messages.append({
                    "role": "assistant",
                    "content": content
                })

                self._total_steps += steps
                self._total_token_usage = self._total_token_usage + total_token_usage

                return AgentResponse(
                    content=content,
                    tool_calls=all_tool_calls,
                    steps=steps,
                    token_usage=total_token_usage
                )

        # Max steps reached
        logger.warning(f"Max steps ({self._max_steps}) reached")
        self._total_steps += steps
        self._total_token_usage = self._total_token_usage + total_token_usage

        return AgentResponse(
            content="Max steps reached without final response",
            tool_calls=all_tool_calls,
            steps=steps,
            token_usage=total_token_usage
        )

    def get_total_steps(self) -> int:
        """Get total steps taken across all runs."""
        return self._total_steps

    def get_total_token_usage(self) -> TokenUsage:
        """Get total token usage across all runs."""
        return self._total_token_usage

    def clear_history(self) -> None:
        """Clear conversation history for a fresh start (keeps system prompt)."""
        self._messages = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})
        self._total_steps = 0
        self._total_token_usage = TokenUsage()


class LitellmAgentFactory(AgentFactory):
    """Factory for creating LiteLLM-based agents.

    This factory creates agents that use JSON function calling
    with a proper agentic loop (execute tools, feed back results).
    """

    def __init__(self, model: LitellmModel, max_steps: int = 20):
        """Initialize the factory with a model configuration.

        Args:
            model: The LiteLLM model configuration
            max_steps: Maximum steps per agent run (default: 20)
        """
        self.model = model
        self.max_steps = max_steps

    def create_agent(
        self,
        functions: List[Callable],
        variables: Optional[List[Any]] = None,
        types: Optional[List[Any]] = None,
        description: Optional[str] = None,
        requirements: Optional[str] = None
    ) -> LitellmAgentWrapper:
        """Create a LiteLLM agent with the specified functions.

        Args:
            functions: List of callable functions
            variables: Ignored (for interface compatibility)
            types: Ignored (for interface compatibility)
            description: Task description for system prompt
            requirements: Task requirements for system prompt

        Returns:
            A LitellmAgentWrapper instance
        """
        return LitellmAgentWrapper(
            model=self.model,
            tools=functions,
            max_steps=self.max_steps,
            description=description,
            requirements=requirements
        )
