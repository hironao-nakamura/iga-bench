from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    raw_response: str
    provider: str
    model_name: str
    temperature: float
    token_usage: dict | None


class BaseProvider(ABC):
    @abstractmethod
    def run(
        self,
        prompt: str,
        *,
        problem: dict | None = None,
        premises: list[dict] | None = None,
        question: str | None = None,
        temperature_override: float | None = None,
    ) -> ProviderResponse:
        raise NotImplementedError
