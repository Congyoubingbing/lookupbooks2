"""
data_builder.py（兼容层）

用于构建 S0（知识框架）。
用法：
    python src/data_builder.py --force
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_SRC = ROOT / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from asa_plus.main import build_s0 as _build_s0


def main():
    force = "--force" in sys.argv
    _build_s0(force=force)


if __name__ == "__main__":
    main()
