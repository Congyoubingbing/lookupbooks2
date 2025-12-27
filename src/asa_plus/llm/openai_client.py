from __future__ import annotations

from typing import List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError

from .base import BaseChatClient, LLMError
from .types import ChatMessage, LLMResult


class OpenAIChatClient(BaseChatClient):
    """
    OpenAI 官方 API（GPT-4o 等）。
    同时也可用于 OpenAI-compatible 的服务（只要 base_url 指向兼容接口）。
    """
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("OpenAIChatClient: api_key 不能为空")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def chat(
        self,
        provider_name: str,
        model: str,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        timeout_s: int,
    ) -> LLMResult:
        try:
            payload = [{"role": m.role, "content": m.content} for m in messages]
            resp: ChatCompletion = self.client.chat.completions.create(
                model=model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
            text = resp.choices[0].message.content or ""
            return LLMResult(provider=provider_name, model=model, text=text, raw=resp)
        except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError) as e:
            raise LLMError(f"[{provider_name}] OpenAI API error: {e}") from e
        except Exception as e:
            raise LLMError(f"[{provider_name}] Unexpected error: {e}") from e
