from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple


class JSONParseError(Exception):
    pass


def _find_json_span(text: str) -> Optional[Tuple[int, int]]:
    """
    在 text 中寻找一个可能的 JSON 对象或数组的起止位置。
    采用括号计数法，支持 { ... } 或 [ ... ]。
    """
    # 找到第一个 { 或 [
    m = re.search(r'[\{\[]', text)
    if not m:
        return None
    start = m.start()
    opener = text[start]
    closer = '}' if opener == '{' else ']'
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return (start, i + 1)

    return None


def extract_json(text: str) -> Any:
    """
    从 LLM 输出中提取 JSON。
    - 若文本中包含 ```json ... ```，优先取 fenced 内容；
    - 否则用括号计数法寻找第一个 JSON 片段。
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        return json.loads(candidate)

    span = _find_json_span(text)
    if not span:
        raise JSONParseError("未找到 JSON 片段")
    candidate = text[span[0]:span[1]].strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # 做一次轻度清洗：去掉尾随注释、去掉多余逗号等（尽量保守）
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise JSONParseError(f"JSON 解析失败: {e}") from e


def to_pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def safe_get(d: Any, path: str, default=None):
    """
    从嵌套 dict 安全取值。path 形如 "a.b.c"
    """
    cur = d
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur
