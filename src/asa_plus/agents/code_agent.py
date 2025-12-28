from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config_loader import AppConfig, resolve_path, ensure_dirs
from ..llm.router import LLMRouter
from ..llm import prompts as P
from ..utils.logger import get_logger

log = get_logger(__name__)


# =========================
# 数据结构定义
# =========================

@dataclass
class GeneratedArtifact:
    """单个生成代码文件的信息"""
    path: str
    abs_path: Path


@dataclass
class CodeGenOutput:
    """代码生成的完整结构化输出"""
    engine_choice: str
    rationale: str
    math_derivation: List[str]
    algorithm: List[str]
    artifacts: List[GeneratedArtifact]
    requirements: List[str]
    run_instructions: List[str]
    expected_outputs: List[str]
    notes_for_user_to_modify: List[str]
    raw_json: Dict[str, Any]


# =========================
# CodeAgent 本体
# =========================

class CodeAgent:
    """
    根据 QuestionAgent 的最终推理结果，生成可运行代码及说明。
    """

    def __init__(self, config: AppConfig, router: Optional[LLMRouter] = None):
        self.config = config
        ensure_dirs(config)
        self.router = router or LLMRouter(config)
        self.out_dir = resolve_path(config, config.paths.generated_code_dir)

    # -------------------------
    # 工具函数：安全路径拼接
    # -------------------------
    @staticmethod
    def _safe_join(base: Path, rel: str) -> Path:
        """
        防止 LLM 生成恶意路径（目录穿越）
        """
        rel_path = Path(rel)
        if rel_path.is_absolute():
            rel_path = Path(rel_path.name)

        candidate = (base / rel_path).resolve()
        if not str(candidate).startswith(str(base.resolve())):
            raise ValueError(f"不安全的输出路径: {rel}")

        return candidate

    # -------------------------
    # 核心接口（⚠️ 系统唯一权威）
    # -------------------------
    def generate(
        self,
        q0: str,
        final_plan: Dict[str, Any],
        used_sources: List[Dict[str, Any]],
        session_id: str,
    ) -> CodeGenOutput:
        """
        生成代码与相关说明

        参数：
        - q0: 原始问题
        - final_plan: QuestionAgent 给出的最终解题规划（结构化）
        - used_sources: 使用到的证据章节
        - session_id: 会话 ID，用于输出隔离
        """

        # 1. 构造 Prompt
        messages = P.prompt_generate_code(
            q0=q0,
            final_plan=final_plan,
            used_sources=used_sources,
        )

        # 2. 调用 LLM（JSON 输出）
        js = self.router.chat_json(
            task="coding",
            messages=messages,
            model_role="coding",
        )

        # 3. 创建会话输出目录
        session_dir = self.out_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # 4. 写出代码文件
        artifacts: List[GeneratedArtifact] = []
        for f in js.get("code_files", []):
            rel = (f.get("path") or "").strip()
            content = f.get("content", "")

            if not rel:
                continue

            abs_path = self._safe_join(session_dir, rel)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")

            artifacts.append(
                GeneratedArtifact(
                    path=rel,
                    abs_path=abs_path,
                )
            )

        # 5. 保存完整原始 JSON（科研可复现）
        (session_dir / "codegen_output.json").write_text(
            json.dumps(js, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 6. 结构化输出
        out = CodeGenOutput(
            engine_choice=js.get("engine_choice", "python"),
            rationale=js.get("rationale", ""),
            math_derivation=js.get("math_derivation", []) or [],
            algorithm=js.get("algorithm", []) or [],
            artifacts=artifacts,
            requirements=js.get("requirements", []) or [],
            run_instructions=js.get("run_instructions", []) or [],
            expected_outputs=js.get("expected_outputs", []) or [],
            notes_for_user_to_modify=js.get("notes_for_user_to_modify", []) or [],
            raw_json=js,
        )

        log.info(
            f"代码生成完成：session={session_id} "
            f"输出文件数={len(artifacts)} dir={session_dir}"
        )

        return out
