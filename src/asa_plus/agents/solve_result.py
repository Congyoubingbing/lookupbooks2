from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Assessment:
    """
    Assessment: 只包含评估相关的字段（我们显式抽取关心的字段）。
    LLM 可能会返回额外字段（例如 used_sources），这些额外字段会被保留在 SolveResult.raw 中。
    """
    can_solve: bool
    confidence: float
    solution_outline: Any  # 结构化方案（可能为列表或 dict）
    conclusions: List[Any]
    solution_steps: List[Any]
    missing_parts: List[Any]
    refine_suggestion: Dict[str, Any]


@dataclass
class SolveResult:
    """
    稳定的 QuestionAgent 返回结构（供 main.py 与其它消费者使用）。

    - plan_text: 人类可读的解题思路（自然语言）
    - evidence_notes: 对 chunk 的证据提取（文本/摘要，用于展示）
    - used_sources: 结构化的证据来源（book/node/chunk），供 CodeAgent/report 使用
    - final_plan: agent 内部的结构化 plan（可直接传给 CodeAgent）
    - session_id: 会话 id（用于输出目录隔离）
    - assessment: Assessment 对象（我们从 dict 中抽取关心字段）
    - raw: 原始 dict（完整保留 LLM 的所有输出，方便 debug/复现）
    """
    plan_text: str
    evidence_notes: List[Any]
    used_sources: List[Dict[str, Any]]
    final_plan: Any
    session_id: str
    assessment: Assessment
    raw: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolveResult":
        # defensive defaults
        plan_text = d.get("plan_text", "")
        evidence_notes = d.get("evidence_notes", []) or []
        used_sources = d.get("used_sources", []) or []
        final_plan = d.get("final_plan", None)
        session_id = d.get("session_id", d.get("session", "")) or d.get("session_id", "")

        # assessment dict must exist; be robust if missing
        assessment_dict = d.get("assessment", {}) or {}

        # extract only fields we care about from assessment (others remain in raw)
        assessment = Assessment(
            can_solve=bool(assessment_dict.get("can_solve", False)),
            confidence=float(assessment_dict.get("confidence", 0.0) or 0.0),
            solution_outline=assessment_dict.get("solution_outline", assessment_dict.get("solution_outline", None)),
            conclusions=assessment_dict.get("conclusions", []) or [],
            solution_steps=assessment_dict.get("solution_steps", []) or assessment_dict.get("solution_outline", []) or [],
            missing_parts=assessment_dict.get("missing_parts", []) or [],
            refine_suggestion=assessment_dict.get("refine_suggestion", {}) or {},
        )

        return cls(
            plan_text=plan_text,
            evidence_notes=evidence_notes,
            used_sources=used_sources,
            final_plan=final_plan,
            session_id=session_id,
            assessment=assessment,
            raw=d,
        )
