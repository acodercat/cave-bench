"""Google Gemini model implementation."""

from typing import Optional, List, Dict, Any, AsyncIterator
from cave_agent import Model
from google.genai import types


class GeminiModel(Model):
    """Gemini model implementation using Google's GenAI SDK."""

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize Gemini model.

        Args:
            model_id: Model identifier
            api_key: API authentication key
            base_url: Optional API endpoint URL (unused for Gemini)
            **kwargs: Additional parameters (e.g., temperature)
        """
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

    async def _prepare_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare parameters for Gemini API call."""
        system_instruction = None
        genai_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "assistant":
                genai_messages.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            elif role == "user":
                genai_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })

        config_dict = {
            "max_output_tokens": 20000,
            "temperature": self.kwargs.get("temperature", 1.0),
        }

        if system_instruction:
            config_dict["system_instruction"] = system_instruction

        config_dict["thinking_config"] = types.ThinkingConfig(thinking_level="low")

        return {
            "contents": genai_messages,
            "config": types.GenerateContentConfig(**config_dict)
        }

    async def call(self, messages: List[Dict[str, str]]) -> str:
        """Generate response with retry logic for malformed responses."""
        from google import genai

        client = genai.Client(api_key=self.api_key)
        params = await self._prepare_params(messages)

        max_retries = 5
        last_response = None

        for attempt in range(max_retries):
            response = await client.aio.models.generate_content(
                model=self.model_id,
                **params
            )
            last_response = response

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason

                if finish_reason and "MALFORMED_FUNCTION_CALL" in str(finish_reason):
                    print(f"Attempt {attempt + 1}/{max_retries}: MALFORMED_FUNCTION_CALL, retrying...")
                    continue

                if candidate.content and candidate.content.parts:
                    content = candidate.content.parts[0].text
                    if content and content.strip():
                        return content
                    print(f"Attempt {attempt + 1}/{max_retries}: Empty content, retrying...")
                    continue

        raise Exception(f"No valid content after {max_retries} attempts: {last_response}")

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens."""
        import litellm

        params = {
            "model": self.model_id,
            "api_key": self.api_key,
            "messages": messages,
            **self.kwargs
        }

        response = await litellm.acompletion(**params, stream=True)

        async for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
