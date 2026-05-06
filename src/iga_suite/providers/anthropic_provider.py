from __future__ import annotations

import os
from anthropic import Anthropic

from iga_suite.providers.base import BaseProvider, ProviderResponse


class AnthropicMessagesProvider(BaseProvider):
    def __init__(self, api_key_env: str, model_name: str, temperature: float = 0.0, max_tokens: int = 1200):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {api_key_env}")
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, prompt: str, *, problem=None, premises=None, question=None, temperature_override: float | None = None) -> ProviderResponse:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature if temperature_override is None else temperature_override,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = response.content[0].text if response.content else ''
        usage = None
        if getattr(response, 'usage', None) is not None:
            usage = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
            }
        return ProviderResponse(
            raw_response=text,
            provider='anthropic',
            model_name=self.model_name,
            temperature=self.temperature if temperature_override is None else temperature_override,
            token_usage=usage,
        )
