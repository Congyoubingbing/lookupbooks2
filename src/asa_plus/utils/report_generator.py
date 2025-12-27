from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config_loader import AppConfig, resolve_path, ensure_dirs
from ..utils.json_utils import to_pretty_json
from ..utils.logger import get_logger


log = get_logger(__name__)


def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else s[:n] + "\n...（截断）..."


class ReportGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        ensure_dirs(config)
        self.reports_dir = resolve_path(config, config.paths.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        session_id: str,
        q0: str,
        s0_outline: str,
        final_plan: Dict[str, Any],
        used_sources: List[Dict[str, Any]],
        codegen_json: Optional[Dict[str, Any]] = None,
        exec_result: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = self.reports_dir / f"report_{session_id}.md"

        lines: List[str] = []
        lines.append(f"# ASA Plus 报告 - session {session_id}\n")
        lines.append("## 问题 Q0\n")
        lines.append(_truncate(q0, 4000) + "\n")

        lines.append("## S0 Outline（用于分类）\n")
        lines.append("```text")
        lines.append(_truncate(s0_outline, 12000))
        lines.append("```\n")

        lines.append("## 多层分类 S1..Sn\n")
        lines.append("```json")
        lines.append(_truncate(to_pretty_json(final_plan.get("classifications", [])), 20000))
        lines.append("```\n")

        lines.append("## 综合评估与解题思路\n")
        lines.append("```json")
        lines.append(_truncate(to_pretty_json(final_plan.get("assessment", {})), 20000))
        lines.append("```\n")

        lines.append("## 使用到的书籍章节（必须可追溯）\n")
        lines.append("```json")
        lines.append(_truncate(to_pretty_json(used_sources), 20000))
        lines.append("```\n")

        if codegen_json:
            lines.append("## 代码生成结果\n")
            lines.append("```json")
            lines.append(_truncate(to_pretty_json(codegen_json), 20000))
            lines.append("```\n")

            if self.config.report.include_full_code_in_report:
                # 把 code_files 内容展开（可能较长）
                files = codegen_json.get("code_files", []) if isinstance(codegen_json, dict) else []
                for f in files:
                    rel = f.get("path", "unknown")
                    content = f.get("content", "")
                    lines.append(f"### 文件: {rel}\n")
                    # 尝试按后缀渲染 code fence
                    lang = "python"
                    if rel.endswith(".in"):
                        lang = "text"
                    elif rel.endswith(".mdp") or rel.endswith(".gro") or rel.endswith(".top"):
                        lang = "text"
                    lines.append(f"```{lang}")
                    lines.append(_truncate(content, 30000))
                    lines.append("```\n")

        if exec_result:
            lines.append("## 执行结果\n")
            lines.append("```json")
            lines.append(_truncate(to_pretty_json(exec_result), 20000))
            lines.append("```\n")

        path.write_text("\n".join(lines), encoding="utf-8")
        log.info(f"报告已生成: {path}")
        return path
