from __future__ import annotations

import os
import time
from openai import OpenAI, BadRequestError, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

from iga_suite.providers.base import BaseProvider, ProviderResponse


class OpenAIChatProvider(BaseProvider):
    def __init__(self, api_key_env: str, model_name: str, temperature: float = 0.0, max_tokens: int = 1200, base_url_env: str | None = None):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {api_key_env}")
        base_url = os.environ.get(base_url_env) if base_url_env else None
        self.request_timeout_s = float(os.environ.get('IGA_PROVIDER_TIMEOUT_SECONDS', '90'))
        self.retry_attempts = int(os.environ.get('IGA_PROVIDER_RETRY_ATTEMPTS', '3'))
        self.retry_backoff_s = float(os.environ.get('IGA_PROVIDER_RETRY_BACKOFF_SECONDS', '2'))
        # Let this layer control retries explicitly for stable audit behavior.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.request_timeout_s, max_retries=0)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, prompt: str, *, problem=None, premises=None, question=None, temperature_override: float | None = None) -> ProviderResponse:
        safe_prompt = prompt.encode('utf-8', errors='replace').decode('utf-8')
        temperature = self.temperature if temperature_override is None else temperature_override
        cleaned = ''.join(ch for ch in safe_prompt if ch == '\n' or ch == '\t' or ord(ch) >= 32)
        last_exc = None
        for attempt in range(1, max(self.retry_attempts, 1) + 1):
            payload = safe_prompt
            if attempt > 1:
                payload = cleaned
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': payload}],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.request_timeout_s,
                )
                break
            except BadRequestError as e:
                # Retry only on JSON body parse style errors.
                if 'could not parse the json body' not in str(e).lower():
                    raise
                last_exc = e
            except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError) as e:
                last_exc = e
            if attempt >= max(self.retry_attempts, 1):
                raise last_exc
            # Exponential backoff to absorb transient provider-side queue spikes.
            time.sleep(self.retry_backoff_s * (2 ** (attempt - 1)))
        else:
            raise RuntimeError('Provider call loop exited unexpectedly')
        text = ''
        choices = getattr(response, 'choices', None)
        if choices:
            first = choices[0]
            message = getattr(first, 'message', None)
            content = getattr(message, 'content', None)
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # OpenAI-compatible providers may emit multipart content blocks.
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        v = item.get('text')
                        if isinstance(v, str):
                            parts.append(v)
                    else:
                        v = getattr(item, 'text', None)
                        if isinstance(v, str):
                            parts.append(v)
                text = '\n'.join(parts)
        if not text:
            text = getattr(response, 'output_text', '') or ''
        usage = None
        if response.usage is not None:
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
        return ProviderResponse(
            raw_response=text,
            provider='openai',
            model_name=self.model_name,
            temperature=temperature,
            token_usage=usage,
        )
