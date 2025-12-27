from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config_loader import AppConfig, resolve_path
from ..utils.cache_store import FileCacheStore
from ..utils.hash_utils import sha256_json
from ..utils.json_utils import extract_json, JSONParseError
from ..utils.logger import get_logger

from .base import BaseChatClient, LLMError
from .dashscope_client import DashScopeQwenClient
from .openai_client import OpenAIChatClient
from .types import ChatMessage, LLMResult


log = get_logger(__name__)


@dataclass
class ProviderRuntime:
    name: str
    client: BaseChatClient
    default_temperature: float
    default_max_tokens: int
    timeout_s: int
    models: Dict[str, str]  # outline/reasoning/coding 等


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if not it:
            continue
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _find_json_object(raw: str) -> str:
    """
    从模型输出中尽量提取第一个 JSON object（{...}）。
    兼容 ```json ... ``` 以及前后混杂解释文字。
    """
    if not raw:
        raise ValueError("空响应，无法解析 JSON")

    # 先尝试抓 ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    # 再抓第一个 { ... }，用简单栈匹配（忽略字符串内部的花括号）
    start = raw.find("{")
    if start < 0:
        raise ValueError("未找到 JSON 片段（缺少 '{'）")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]

    raise ValueError("未找到完整 JSON object（花括号不匹配）")


def _repair_invalid_escapes(js: str) -> str:
    r"""
    修复 LLM 常见 JSON 问题（“尽量不破坏结构，只修复字符串内部”）：
    1) 字符串里出现了“裸换行/裸控制字符” -> 转义为 \n / \u00xx
    2) 出现非法反斜杠转义（例如 \l、\u 后面不是 4 个 hex） -> 将反斜杠转义为 \\

    ⚠️ 重要：绝不能把 JSON 结构里的换行（键值之间的格式化换行）替换成 \n，
       否则会导致 `{ \n "a": ... }` 变成 `{\n"a":...}`（\n 不在字符串内）而解析失败。

    注意：这里使用 raw string 避免 Windows 下 `\uXXXX` 触发 unicodeescape。
    """
    # 统一换行
    js = js.replace("\r\n", "\n").replace("\r", "\n")
    # 统一中文引号
    js = js.replace("“", '"').replace("”", '"')

    out: List[str] = []
    in_str = False
    i = 0
    while i < len(js):
        ch = js[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue

        # in_str == True
        if ch == '"':
            out.append(ch)
            in_str = False
            i += 1
            continue

        # 处理反斜杠
        if ch == "\\":
            if i + 1 >= len(js):
                out.append("\\\\")
                i += 1
                continue

            nxt = js[i + 1]

            # 有些 latex 指令以 \t \n \r \b \f 开头（例如 \theta, \nabla, \rangle）
            # JSON 会把它们当成 tab/newline/CR 等转义，语义会被破坏；
            # 这里做一个启发式：若 \t/\n/\r/\b/\f 后面紧跟字母，优先当作 latex，转义反斜杠本身。
            if nxt in ("b", "f", "n", "r", "t") and (i + 2 < len(js)) and js[i + 2].isalpha():
                out.append("\\\\")
                i += 1
                continue

            if nxt in ('"', "\\", "/", "b", "f", "n", "r", "t"):
                out.append("\\")
                out.append(nxt)
                i += 2
                continue

            if nxt == "u":
                # 要求后面 4 个 hex
                if i + 6 <= len(js) and re.match(r"[0-9a-fA-F]{4}", js[i + 2 : i + 6]):
                    out.append("\\")
                    out.append("u")
                    out.append(js[i + 2 : i + 6])
                    i += 6
                    continue
                # 非法 \u -> 转义反斜杠
                out.append("\\\\")
                i += 1
                continue

            # 其他非法 escape，例如 \l, \q -> 转义反斜杠
            out.append("\\\\")
            i += 1
            continue

        # 裸控制字符（JSON 字符串里不允许）
        if ch == "\n":
            out.append("\\n")
            i += 1
            continue
        if ch == "\t":
            out.append("\\t")
            i += 1
            continue
        if ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
            i += 1
            continue

        out.append(ch)
        i += 1

    repaired = "".join(out)
    # 去掉结构末尾常见的 trailing comma（尽量保守）
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _loads_json_robust(raw: str) -> Any:
    """
    强健 JSON 解析：
    1) 优先用项目的 extract_json（通常能处理 ```json ```）
    2) 若失败：手动提取 {...} 并 json.loads
       - 若 JSONDecodeError：做 escape/控制字符修复后再 loads
    """
    try:
        return extract_json(raw)
    except Exception:
        js = _find_json_object(raw)
        try:
            return json.loads(js)
        except json.JSONDecodeError:
            repaired = _repair_invalid_escapes(js)
            return json.loads(repaired)


class LLMRouter:
    """
    统一路由：按 task -> provider_priority 选择 provider，并失败自动 fallback。
    强制“全部走 API”，不允许本地模型。
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.providers: Dict[str, ProviderRuntime] = {}
        self.cache: Optional[FileCacheStore] = None

        if config.llm.use_cache:
            cache_dir = resolve_path(config, config.paths.cache_dir)
            self.cache = FileCacheStore(cache_dir, ttl_days=config.llm.cache_ttl_days)

        self._init_providers()

    def _get_env(self, key: str) -> str:
        return os.getenv(key, "").strip()

    def _init_providers(self) -> None:
        for name, pconf in self.config.llm.providers.items():
            api_key = self._get_env(pconf.api_key_env)
            if not api_key:
                raise ValueError(f"环境变量 {pconf.api_key_env} 未设置，无法初始化 provider={name}")

            if pconf.type in ("openai", "openai_compatible"):
                base_url = None
                if pconf.base_url_env:
                    base_url = self._get_env(pconf.base_url_env) or None

                # 默认 base_url
                if pconf.type == "openai" and base_url is None:
                    base_url = "https://api.openai.com/v1"
                if pconf.type == "openai_compatible" and base_url is None:
                    # DeepSeek 默认
                    base_url = "https://api.deepseek.com/v1"

                client = OpenAIChatClient(api_key=api_key, base_url=base_url)

            elif pconf.type == "dashscope":
                client = DashScopeQwenClient(api_key=api_key)

            else:
                raise ValueError(f"未知 provider type: {pconf.type}")

            self.providers[name] = ProviderRuntime(
                name=name,
                client=client,
                default_temperature=pconf.default_temperature,
                default_max_tokens=pconf.default_max_tokens,
                timeout_s=pconf.timeout_s,
                models=pconf.models,
            )

    def _cache_key(
        self,
        provider: str,
        model: str,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        return sha256_json(payload)

    @retry(
        retry=retry_if_exception_type(LLMError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _call_once(
        self,
        prt: ProviderRuntime,
        model: str,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        return prt.client.chat(
            provider_name=prt.name,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=prt.timeout_s,
        )

    def _provider_order_for_task(self, task: str) -> List[str]:
        """
        计算某个 task 的 provider 顺序：
        - 若 llm.routing.<task>.provider_priority 非空：使用它
        - 否则：使用 llm.default_provider_priority   ★新增
        - 若两者都为空：使用 providers 的声明顺序（不推荐，最好显式配置）
        """
        rule = self.config.llm.routing.get(task)
        if not rule:
            raise ValueError(f"未配置 llm.routing.{task}")

        order = list(rule.provider_priority) if rule.provider_priority else list(self.config.llm.default_provider_priority)
        if not order:
            order = list(self.config.llm.providers.keys())

        order = _dedupe_keep_order(order)
        order = [p for p in order if p in self.providers]

        if not order:
            raise ValueError(
                f"task={task} 没有可用的 provider。请检查 llm.routing.{task}.provider_priority 或 llm.default_provider_priority"
            )
        return order

    def chat(
        self,
        task: str,
        messages: List[ChatMessage],
        model_role: str = "reasoning",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_debug: Optional[dict] = None,
    ) -> LLMResult:
        """
        task: routing key，如 s0_outline/decomposition/content_evidence/integration/coding
        model_role: outline/reasoning/coding (从 providers.*.models 中取)
        """
        provider_order = self._provider_order_for_task(task)

        last_err: Optional[Exception] = None
        for provider_name in provider_order:
            prt = self.providers.get(provider_name)
            if not prt:
                continue

            model = prt.models.get(model_role) or prt.models.get("reasoning")
            if not model:
                last_err = ValueError(f"provider={provider_name} 未配置 models[{model_role}]")
                continue

            temp = temperature if temperature is not None else prt.default_temperature
            mt = max_tokens if max_tokens is not None else prt.default_max_tokens

            ck = self._cache_key(provider_name, model, messages, temp, mt)
            if self.cache:
                item = self.cache.get(ck)
                if item is not None and isinstance(item.value, str):
                    return LLMResult(provider=provider_name, model=model, text=item.value, raw=None)

            try:
                res = self._call_once(prt, model, messages, temp, mt)
                if self.cache:
                    self.cache.set(
                        ck,
                        res.text,
                        meta={"task": task, "model_role": model_role, "extra_debug": extra_debug or {}},
                    )
                return res
            except Exception as e:
                last_err = e
                log.warning(f"LLM 调用失败，provider={provider_name}, model={model}, err={e}")

        raise LLMError(f"所有 provider 都失败了 (task={task}). last_err={last_err}")

    def chat_json(
        self,
        task: str,
        messages: List[ChatMessage],
        model_role: str = "reasoning",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        res = self.chat(task, messages, model_role=model_role, temperature=temperature, max_tokens=max_tokens)
        try:
            return _loads_json_robust(res.text)
        except Exception as e:
            preview = (res.text or "")[:2400]
            log.error(f"JSON 解析失败: {e}\n--- RAW ---\n{preview}\n--- END ---")
            raise