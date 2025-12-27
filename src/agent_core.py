"""
agent_core.py（兼容层）

你原项目里已有 agent_core.py / data_builder.py / llm_client.py。
为了尽量减少你迁移时的改动，本文件保留同名入口，但内部调用新的 asa_plus 包实现。

用法：
    python src/agent_core.py "你的问题Q0"

等价于：
    python main.py ask "你的问题Q0"
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目的 src/ 加入 sys.path，保证 asa_plus 可 import
ROOT = Path(__file__).resolve().parents[1]  # lookupbooks/
PKG_SRC = ROOT / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from asa_plus.main import ask as _ask


def main():
    q0 = sys.argv[1] if len(sys.argv) > 1 else None
    # 直接复用 Typer command 函数
    _ask(question=q0, no_exec=False, non_interactive=False)


if __name__ == "__main__":
    main()
