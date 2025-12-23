"""Anthropic Claude model implementation."""

from typing import List, Dict, Optional, Any, AsyncIterator
from cave_agent.models import Model


class AnthropicModel(Model):
    """Anthropic Claude model implementation with prompt caching."""

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize Anthropic model.

        Args:
            model_id: Model identifier
            api_key: API authentication key
            base_url: Optional API endpoint URL
            **kwargs: Additional parameters (e.g., temperature, max_tokens)
        """
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

    def _prepare_messages(self, messages: List[Dict[str, str]]):
        """Separate system and conversation messages."""
        system_messages = []
        conversation_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_messages.append({
                    "type": "text",
                    "text": msg["content"],
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                })
            else:
                conversation_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return system_messages, conversation_messages

    async def call(self, messages: List[Dict[str, str]]) -> str:
        """Generate response."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        system_messages, conversation_messages = self._prepare_messages(messages)

        request_params = {
            "model": self.model_id,
            "messages": conversation_messages,
            "temperature": self.kwargs.get("temperature", 0.2),
            "max_tokens": self.kwargs.get("max_tokens", 12000),
        }

        if system_messages:
            request_params["system"] = system_messages

        response = await client.messages.create(**request_params)
        return response.content[0].text

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        system_messages, conversation_messages = self._prepare_messages(messages)

        request_params = {
            "model": self.model_id,
            "messages": conversation_messages,
            "temperature": self.kwargs.get("temperature", 0.2),
            "max_tokens": self.kwargs.get("max_tokens", 12000),
        }

        if system_messages:
            request_params["system"] = system_messages

        async with client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text
