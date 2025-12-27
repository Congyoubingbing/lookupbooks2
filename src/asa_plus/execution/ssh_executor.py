from __future__ import annotations

import os
import posixpath
from pathlib import Path
from typing import Dict, Any, Tuple

import paramiko

from .executor import BaseExecutor, ExecResult
from ..config_loader import AppConfig
from ..utils.logger import get_logger


log = get_logger(__name__)


def _sftp_mkdirs(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    parts = []
    p = remote_path
    while p and p not in ("/", ""):
        parts.append(p)
        p = posixpath.dirname(p)
    for d in reversed(parts):
        try:
            sftp.stat(d)
        except FileNotFoundError:
            try:
                sftp.mkdir(d)
            except Exception:
                # 并发情况下可能已经创建
                pass


def _upload_dir(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str) -> None:
    _sftp_mkdirs(sftp, remote_dir)
    for item in local_dir.iterdir():
        if item.is_dir():
            _upload_dir(sftp, item, posixpath.join(remote_dir, item.name))
        else:
            remote_file = posixpath.join(remote_dir, item.name)
            sftp.put(str(item), remote_file)


class SSHExecutor(BaseExecutor):
    def run(self, config: AppConfig, workdir: str, entrypoint: str, timeout_s: int = 3600) -> ExecResult:
        ssh_conf = config.execution.remote_ssh
        host = ssh_conf.host
        port = ssh_conf.port
        username = ssh_conf.username
        password = ssh_conf.password or None
        key_path = ssh_conf.key_path or None
        remote_base = ssh_conf.workdir.rstrip("/")
        python_bin = ssh_conf.python_bin

        local_wd = Path(workdir).resolve()
        ep = Path(entrypoint).resolve()
        if not ep.exists():
            return ExecResult(False, "", f"entrypoint not found: {ep}", 2, {}, {"mode":"remote_ssh"})

        # remote dir named by local workdir name
        remote_dir = posixpath.join(remote_base, local_wd.name)
        rel_ep = os.path.relpath(str(ep), str(local_wd)).replace("\\", "/")
        remote_ep = posixpath.join(remote_dir, rel_ep)

        log.info(f"SSH 连接: {username}@{host}:{port}")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            if key_path:
                pkey = paramiko.RSAKey.from_private_key_file(key_path)
                client.connect(hostname=host, port=port, username=username, pkey=pkey, timeout=20)
            else:
                client.connect(hostname=host, port=port, username=username, password=password, timeout=20)

            sftp = client.open_sftp()
            log.info(f"上传目录到远程: {local_wd} -> {remote_dir}")
            _upload_dir(sftp, local_wd, remote_dir)
            sftp.close()

            cmd = f"cd {remote_dir} && {python_bin} {remote_ep}"
            log.info(f"远程执行: {cmd}")

            stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout_s)
            out = stdout.read().decode("utf-8", errors="ignore")
            err = stderr.read().decode("utf-8", errors="ignore")
            rc = stdout.channel.recv_exit_status()
            ok = rc == 0
            return ExecResult(
                ok=ok,
                stdout=out,
                stderr=err,
                return_code=rc,
                artifacts={},
                meta={"mode":"remote_ssh","remote_dir":remote_dir,"cmd":cmd},
            )
        except Exception as e:
            return ExecResult(
                ok=False,
                stdout="",
                stderr=str(e),
                return_code=1,
                artifacts={},
                meta={"mode":"remote_ssh"},
            )
        finally:
            try:
                client.close()
            except Exception:
                pass
