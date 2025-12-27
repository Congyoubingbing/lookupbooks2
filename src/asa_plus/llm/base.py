from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .types import ChatMessage, LLMResult


class LLMError(Exception):
    pass


class BaseChatClient(ABC):
    @abstractmethod
    def chat(
        self,
        provider_name: str,
        model: str,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        timeout_s: int,
    ) -> LLMResult:
        raise NotImplementedError
