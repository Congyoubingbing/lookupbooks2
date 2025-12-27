from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class CacheItem:
    key: str
    created_at: float
    value: Any
    meta: dict


class FileCacheStore:
    """
    简单文件缓存：每个 key 一个 json 文件。
    """
    def __init__(self, cache_dir: Path, ttl_days: int = 365):
        self.cache_dir = cache_dir
        self.ttl_s = ttl_days * 24 * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[CacheItem]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            created_at = float(data.get("created_at", 0))
            if self.ttl_s > 0 and (time.time() - created_at) > self.ttl_s:
                # 过期
                try:
                    p.unlink()
                except Exception:
                    pass
                return None
            return CacheItem(
                key=key,
                created_at=created_at,
                value=data.get("value"),
                meta=data.get("meta", {}),
            )
        except Exception:
            return None

    def set(self, key: str, value: Any, meta: Optional[dict] = None) -> None:
        p = self._path(key)
        payload = {
            "created_at": time.time(),
            "value": value,
            "meta": meta or {},
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
