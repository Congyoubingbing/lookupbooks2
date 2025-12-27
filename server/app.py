from __future__ import annotations

"""
一个最小可用的远程执行服务端示例（配合 execution.remote_http 使用）。

启动：
    pip install -r server/requirements.txt
    uvicorn server.app:app --host 0.0.0.0 --port 8000

客户端在 config/config.yaml 中设置：
    execution:
      mode: "remote_http"
      remote_http:
        endpoint: "http://<server-ip>:8000/api/run"
        token: ""  # 可选

安全提示：
- 这是示例实现，默认不做沙箱隔离；
- 强烈建议在容器/隔离环境中运行，并增加鉴权、资源限制、白名单等。
"""

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


class RunRequest(BaseModel):
    entrypoint: str
    workdir_name: str
    zip_b64: str
    timeout_s: int = 3600


class RunResponse(BaseModel):
    ok: bool
    stdout: str
    stderr: str
    return_code: int
    artifacts: Dict[str, str] = {}
    meta: Dict[str, Any] = {}


app = FastAPI(title="ASAPlus Remote Runner")


def _safe_extract_zip(zip_bytes: bytes, dst_dir: str) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for m in zf.infolist():
            # 防 zip slip
            if ".." in m.filename.replace("\\", "/"):
                raise ValueError("zip contains illegal path")
        zf.extractall(dst_dir)


@app.post("/api/run", response_model=RunResponse)
def run(req: RunRequest, authorization: Optional[str] = Header(default=None)):
    # TODO: 如果需要鉴权，可在这里校验 authorization
    # if authorization != "Bearer <TOKEN>": raise HTTPException(401, "unauthorized")

    try:
        zip_bytes = base64.b64decode(req.zip_b64.encode("ascii"))
    except Exception:
        raise HTTPException(400, "invalid zip_b64")

    with tempfile.TemporaryDirectory(prefix="asa_plus_") as td:
        workdir = os.path.join(td, req.workdir_name)
        os.makedirs(workdir, exist_ok=True)

        try:
            _safe_extract_zip(zip_bytes, workdir)
        except Exception as e:
            raise HTTPException(400, f"bad zip: {e}")

        ep = os.path.normpath(os.path.join(workdir, req.entrypoint))
        if not ep.startswith(workdir):
            raise HTTPException(400, "illegal entrypoint path")
        if not os.path.exists(ep):
            raise HTTPException(400, f"entrypoint not found: {req.entrypoint}")

        cmd = ["python3", ep]
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=req.timeout_s,
                env=os.environ.copy(),
            )
            ok = proc.returncode == 0
            return RunResponse(
                ok=ok,
                stdout=proc.stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
                artifacts={},
                meta={"cmd": cmd, "cwd": workdir},
            )
        except subprocess.TimeoutExpired as e:
            return RunResponse(
                ok=False,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + f"\n[timeout] {req.timeout_s}s",
                return_code=124,
                artifacts={},
                meta={"cmd": cmd, "cwd": workdir},
            )
