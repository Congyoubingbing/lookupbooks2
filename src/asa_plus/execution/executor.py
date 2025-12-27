from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Literal

from ..config_loader import AppConfig


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    return_code: int
    artifacts: Dict[str, str]
    meta: Dict[str, Any]


class BaseExecutor:
    def run(self, config: AppConfig, workdir: str, entrypoint: str, timeout_s: int = 3600) -> ExecResult:
        raise NotImplementedError


def select_executor(config: AppConfig) -> BaseExecutor:
    mode = config.execution.mode
    if mode == "local":
        from .local_executor import LocalExecutor
        return LocalExecutor()
    if mode == "remote_ssh":
        from .ssh_executor import SSHExecutor
        return SSHExecutor()
    if mode == "remote_http":
        from .http_executor import HTTPExecutor
        return HTTPExecutor()
    raise ValueError(f"未知 execution.mode: {mode}")
