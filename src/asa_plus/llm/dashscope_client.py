from __future__ import annotations

from typing import List

import dashscope
from http import HTTPStatus

from .base import BaseChatClient, LLMError
from .types import ChatMessage, LLMResult


class DashScopeQwenClient(BaseChatClient):
    """
    阿里云 DashScope（Qwen 系列）API 客户端。
    不需要本地部署模型。
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("DashScopeQwenClient: api_key 不能为空")
        dashscope.api_key = api_key

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
            resp = dashscope.Generation.call(
                model=model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                # DashScope 的 timeout 参数支持与否取决于 SDK 版本；若不支持将被忽略
                timeout=timeout_s,
            )
            if resp.status_code != HTTPStatus.OK:
                raise LLMError(f"[{provider_name}] DashScope error: {resp.code} {resp.message}")
            text = resp.output.choices[0]["message"]["content"]
            return LLMResult(provider=provider_name, model=model, text=text, raw=resp)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"[{provider_name}] DashScope unexpected error: {e}") from e
