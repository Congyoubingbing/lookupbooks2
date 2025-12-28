from __future__ import annotations

import sys
from pathlib import Path

# 让 src/ 可被 import
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asa_plus.main import main

if __name__ == "__main__":
    main()
