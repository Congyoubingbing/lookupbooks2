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


@dataclass
class GeneratedArtifact:
    path: str
    abs_path: Path


@dataclass
class CodeGenOutput:
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


class CodeAgent:
    def __init__(self, config: AppConfig, router: Optional[LLMRouter] = None):
        self.config = config
        ensure_dirs(config)
        self.router = router or LLMRouter(config)
        self.out_dir = resolve_path(config, config.paths.generated_code_dir)

    @staticmethod
    def _safe_join(base: Path, rel: str) -> Path:
        # 防目录穿越
        rel_path = Path(rel)
        if rel_path.is_absolute():
            # 强制转为相对路径
            rel_path = Path(rel_path.name)
        candidate = (base / rel_path).resolve()
        if not str(candidate).startswith(str(base.resolve())):
            raise ValueError(f"不安全的输出路径: {rel}")
        return candidate

    def generate(self, q0: str, final_plan: Dict[str, Any], used_sources: List[Dict[str, Any]], session_id: str) -> CodeGenOutput:
        msgs = P.prompt_generate_code(q0=q0, final_plan=final_plan, used_sources=used_sources)
        js = self.router.chat_json(task="coding", messages=msgs, model_role="coding")

        session_dir = self.out_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        artifacts: List[GeneratedArtifact] = []
        for f in js.get("code_files", []):
            rel = f.get("path", "").strip()
            content = f.get("content", "")
            if not rel:
                continue
            abs_path = self._safe_join(session_dir, rel)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
            artifacts.append(GeneratedArtifact(path=rel, abs_path=abs_path))

        # 额外保存一份原始 JSON
        (session_dir / "codegen_output.json").write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")

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
        log.info(f"代码生成完成：session={session_id} 输出文件数={len(artifacts)} dir={session_dir}")
        return out
