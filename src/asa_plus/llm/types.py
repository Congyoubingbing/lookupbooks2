from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Any


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass
class LLMCall:
    provider: str
    model: str
    messages: List[ChatMessage]
    temperature: float
    max_tokens: int
    timeout_s: int
    extra: Dict[str, Any]


@dataclass
class LLMResult:
    provider: str
    model: str
    text: str
    raw: Optional[Any] = None
