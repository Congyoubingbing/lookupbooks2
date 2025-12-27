from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Any

from .executor import BaseExecutor, ExecResult
from ..config_loader import AppConfig
from ..utils.logger import get_logger


log = get_logger(__name__)


class LocalExecutor(BaseExecutor):
    def run(self, config: AppConfig, workdir: str, entrypoint: str, timeout_s: int = 3600) -> ExecResult:
        py = config.execution.local.python_bin
        wd = Path(workdir).resolve()
        wd.mkdir(parents=True, exist_ok=True)

        ep = Path(entrypoint).resolve()
        if not ep.exists():
            return ExecResult(
                ok=False,
                stdout="",
                stderr=f"entrypoint not found: {ep}",
                return_code=2,
                artifacts={},
                meta={"mode": "local"},
            )

        cmd = [py, str(ep)]
        log.info(f"本地执行: {' '.join(cmd)} (cwd={wd})")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(wd),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=os.environ.copy(),
            )
            ok = proc.returncode == 0
            return ExecResult(
                ok=ok,
                stdout=proc.stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
                artifacts={},
                meta={"mode": "local", "cwd": str(wd), "cmd": cmd},
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                ok=False,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + f"\n[timeout] {timeout_s}s",
                return_code=124,
                artifacts={},
                meta={"mode": "local", "cwd": str(wd), "cmd": cmd},
            )
