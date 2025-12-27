from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    dumped = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return sha256_text(dumped)
