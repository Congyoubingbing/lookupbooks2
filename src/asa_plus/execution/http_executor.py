from __future__ import annotations

import base64
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, Any

import requests

from .executor import BaseExecutor, ExecResult
from ..config_loader import AppConfig
from ..utils.logger import get_logger


log = get_logger(__name__)


def _zip_dir_to_bytes(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                arcname = str(p.relative_to(root)).replace("\\", "/")
                zf.write(str(p), arcname)
    return buf.getvalue()


class HTTPExecutor(BaseExecutor):
    """
    远程 HTTP 执行器（需要你自己在服务器实现 endpoint）。
    请求体包含：zip 压缩的工作目录 + entrypoint 相对路径。
    """
    def run(self, config: AppConfig, workdir: str, entrypoint: str, timeout_s: int = 3600) -> ExecResult:
        http_conf = config.execution.remote_http
        endpoint = http_conf.endpoint
        token = http_conf.token

        local_wd = Path(workdir).resolve()
        ep = Path(entrypoint).resolve()
        if not ep.exists():
            return ExecResult(False, "", f"entrypoint not found: {ep}", 2, {}, {"mode":"remote_http"})

        rel_ep = os.path.relpath(str(ep), str(local_wd)).replace("\\", "/")
        zip_bytes = _zip_dir_to_bytes(local_wd)
        zip_b64 = base64.b64encode(zip_bytes).decode("ascii")

        payload = {
            "entrypoint": rel_ep,
            "workdir_name": local_wd.name,
            "zip_b64": zip_b64,
            "timeout_s": timeout_s,
        }
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        log.info(f"HTTP 执行请求: {endpoint}")
        try:
            resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=timeout_s)
            if resp.status_code != 200:
                return ExecResult(False, "", f"HTTP {resp.status_code}: {resp.text[:500]}", resp.status_code, {}, {"mode":"remote_http"})
            data = resp.json()
            return ExecResult(
                ok=bool(data.get("ok", False)),
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
                return_code=int(data.get("return_code", 1)),
                artifacts=data.get("artifacts", {}) or {},
                meta={"mode":"remote_http","endpoint":endpoint},
            )
        except Exception as e:
            return ExecResult(False, "", str(e), 1, {}, {"mode":"remote_http"})
