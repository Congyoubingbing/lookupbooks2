"""
llm_client.py（兼容层）

旧版可能只有 Qwen (DashScope)。
新版实现为 LLMRouter：统一路由到 OpenAI / Qwen / DeepSeek（全部 API）。

这里保留 create_llm_client() 形式，方便你旧代码迁移。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_SRC = ROOT / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from asa_plus.config_loader import load_config
from asa_plus.llm.router import LLMRouter


def create_llm_client():
    config = load_config()
    return LLMRouter(config)
