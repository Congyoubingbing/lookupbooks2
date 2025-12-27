from __future__ import annotations

from typing import Any, Dict, List, Optional

from .types import ChatMessage
from ..utils.json_utils import to_pretty_json


def _json_only_instruction() -> str:
    # 强约束：减少“Invalid \escape / 裸换行 / 截断”
    return (
        "你必须只输出【严格 JSON】(RFC8259)：\n"
        "1) 只能输出一个 JSON 对象，必须以 { 开始，以 } 结束。\n"
        "2) 不要输出 Markdown 代码块，不要输出任何解释文字。\n"
        "3) JSON 字符串中禁止出现原始换行；如需换行必须写成 \\\\n。\n"
        "4) JSON 字符串中所有反斜杠 \\ 必须写成 \\\\（例如 LaTeX 的 \\theta 必须写成 \\\\theta）。\n"
        "5) 不要输出未闭合的数组/对象。\n"
        "6) 为避免输出过长被截断：key_points 最多 12 条，key_concepts 最多 20 条，formulas 最多 8 条，cross_links 最多 8 条。\n"
        "若无法确定某字段，请给出空数组或空字符串，但 JSON 结构必须完整可解析。"
    )


def prompt_s0_summarize_node(book: Dict[str, Any], node: Dict[str, Any], content: str) -> List[ChatMessage]:
    system = (
        "你是高分子物理/流变学/分子模拟领域的科研助手，擅长从原始教材中提炼结构化知识框架。"
        "你的任务是：对给定节点（章/节/小节）的原文内容做结构化梳理。"
    )
    user = {
        "task": "S0_NODE_SUMMARY",
        "book": {
            "book_id": book.get("book_id"),
            "book_name": book.get("book_name"),
        },
        "node": {
            "node_id": node.get("node_id"),
            "level": node.get("level"),
            "title": node.get("title"),
            "path_titles": node.get("path_titles", []),
        },
        "requirements": {
            "output_language": "zh",
            "must_include_source": True,
            "must_extract_key_concepts": True,
            "must_extract_formulas": True,
        },
        "output_schema": {
            "book_id": "string",
            "node_id": "string",
            "level": "int",
            "title": "string",
            "summary": "string",
            "key_points": ["string (<=12)"],
            "key_concepts": ["string (<=20)"],
            "formulas": [
                {"latex": "string", "meaning": "string", "variables": [{"symbol":"string","meaning":"string"}]}
            ],
            "cross_links": [
                {"related_node_id":"string","relation":"string","note":"string"}
            ],
            "source": {"book_name":"string","node_path":"string"}
        },
        "text": content,
        "json_only": True
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_json_only_instruction() + "\n\n" + to_pretty_json(user)),
    ]


def prompt_s0_summarize_node_short(book: Dict[str, Any], node: Dict[str, Any], content: str) -> List[ChatMessage]:
    """
    短版：用于解析失败/输出过长时重试，减少截断概率。
    """
    system = (
        "你是高分子物理/流变学/分子模拟领域的科研助手。"
        "请对该节点做简明结构化摘要（短输出），只保留最关键的信息。"
    )
    user = {
        "task": "S0_NODE_SUMMARY_SHORT",
        "book": {"book_id": book.get("book_id"), "book_name": book.get("book_name")},
        "node": {
            "node_id": node.get("node_id"),
            "level": node.get("level"),
            "title": node.get("title"),
            "path_titles": node.get("path_titles", []),
        },
        "output_schema": {
            "book_id": "string",
            "node_id": "string",
            "level": "int",
            "title": "string",
            "summary": "string",
            "key_points": ["string (<=8)"],
            "key_concepts": ["string (<=12)"],
            "source": {"book_name":"string","node_path":"string"}
        },
        "text": content,
        "json_only": True
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_json_only_instruction() + "\n\n" + to_pretty_json(user)),
    ]


def prompt_decompose_question(
    q0: str,
    s0_outline: str,
    depth: int,
    max_selected_nodes: int,
    max_subquestions: int,
    prev_classification: Optional[Dict[str, Any]] = None,
) -> List[ChatMessage]:
    system = (
        "你是高分子物理/流变学/分子模拟专家，擅长把复杂科研问题拆解为可解的子问题并映射到教材章节。"
        "你会收到：用户问题 Q0，以及知识框架 S0（仅包含 node_id + 章节/小节标题）。"
        "你的任务是：给出当前层级 depth 的分类结果 S_depth。"
    )

    payload = {
        "task": "DECOMPOSE_Q0",
        "depth": depth,
        "question": q0,
        "s0_outline": s0_outline,
        "constraints": {
            "max_selected_nodes": max_selected_nodes,
            "max_subquestions": max_subquestions,
            "note": "selected_node_ids 必须来自 s0_outline 中的 node_id；不要杜撰不存在的 id。",
        },
        "output_schema": {
            "depth": "int",
            "selected_nodes": [
                {"node_id": "string", "why_relevant": "string", "priority": "int"}
            ],
            "subquestions": [
                {
                    "sub_id": "string",
                    "question": "string",
                    "goal": "string",
                    "expected_output": "string",
                    "related_node_ids": ["string"]
                }
            ],
            "confidence": "float (0~1)",
            "need_more_detail": "bool",
            "notes": "string"
        },
        "json_only": True
    }

    msgs = [ChatMessage(role="system", content=system)]
    if prev_classification:
        msgs.append(ChatMessage(role="user", content="上一层分类结果 prev_S:\n" + to_pretty_json(prev_classification)))

    msgs.append(ChatMessage(role="user", content=_json_only_instruction() + "\n\n" + to_pretty_json(payload)))
    return msgs


def prompt_extract_evidence_from_chunk(
    q0: str,
    depth: int,
    node_meta: Dict[str, Any],
    chunk_text: str,
    chunk_id: str,
    chunk_index: int,
    chunk_total: int,
) -> List[ChatMessage]:
    system = (
        "你是严谨的科研助手。你将读取教材原文片段（chunk），"
        "只提取与用户问题相关的事实、概念、定义、公式、推导步骤和可用于建模/编程的要点。"
        "注意：你必须在输出中标注来源 node_id。"
    )
    payload = {
        "task": "EVIDENCE_EXTRACTION",
        "depth": depth,
        "question": q0,
        "source": {
            "book_id": node_meta.get("book_id"),
            "book_name": node_meta.get("book_name"),
            "node_id": node_meta.get("node_id"),
            "node_title": node_meta.get("title"),
            "node_path": node_meta.get("path_str"),
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "chunk_total": chunk_total
        },
        "output_schema": {
            "node_id": "string",
            "chunk_id": "string",
            "relevant_points": ["string"],
            "relevant_formulas": [
                {"latex":"string","meaning":"string","variables":[{"symbol":"string","meaning":"string"}]}
            ],
            "assumptions_or_conditions": ["string"],
            "direct_quotes": [
                {"quote":"string (<=200字，尽量原文)", "why":"string"}
            ]
        },
        "text": chunk_text,
        "json_only": True
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_json_only_instruction() + "\n\n" + to_pretty_json(payload))
    ]


def prompt_integrate_evidence(
    q0: str,
    depth: int,
    classification: Dict[str, Any],
    s0_outline: str,
    evidence_notes: List[Dict[str, Any]],
) -> List[ChatMessage]:
    system = (
        "你是高分子物理/流变学/分子模拟专家。你将获得：Q0、当前分类 S_depth、以及从选中章节原文逐块提取的 evidence notes。"
        "你的任务：\n"
        "1) 基于 evidence notes 做严谨推理，给出解题思路（可包含必要公式推导框架）。\n"
        "2) 判断是否已经可以解决 Q0（can_solve）。\n"
        "3) 无论能否解决，都必须列出 used_sources（明确到 book + node_id + 章节路径）。\n"
        "4) 若不能解决，指出 missing_parts，并提出 refine_suggestion：需要进一步细分到哪些 node 或需要哪些额外信息。"
    )
    payload = {
        "task": "INTEGRATE_AND_ASSESS",
        "depth": depth,
        "question": q0,
        "S_depth": classification,
        "s0_outline": s0_outline,
        "evidence_notes": evidence_notes,
        "output_schema": {
            "can_solve": "bool",
            "confidence": "float (0~1)",
            "solution_outline": [
                {"step":"string", "detail":"string", "equations":["string"], "notes":"string"}
            ],
            "used_sources": [
                {"book_name":"string","book_id":"string","node_id":"string","node_path":"string","how_used":"string"}
            ],
            "conclusions": ["string"],
            "missing_parts": ["string"],
            "refine_suggestion": {
                "need_deeper_nodes": ["string (node_id)"],
                "need_user_inputs": ["string"],
                "reason": "string"
            }
        },
        "json_only": True
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_json_only_instruction() + "\n\n" + to_pretty_json(payload))
    ]


def prompt_refine_classification(
    q0: str,
    depth_next: int,
    s0_outline: str,
    last_classification: Dict[str, Any],
    last_assessment: Dict[str, Any],
    max_selected_nodes: int,
    max_subquestions: int,
) -> List[ChatMessage]:
    system = (
        "你是问题分解与知识映射专家。你将收到：Q0、S0 outline、上一层分类 S_prev，以及上一轮综合评估结果 assessment（包含 missing_parts 与 refine_suggestion）。"
        "你的任务：生成更细的一层分类 S_next。"
        "必须使用 s0_outline 中存在的 node_id。"
    )
    payload = {
        "task": "REFINE_CLASSIFICATION",
        "depth": depth_next,
        "question": q0,
        "s0_outline": s0_outline,
        "S_prev": last_classification,
        "assessment_prev": last_assessment,
        "constraints": {
            "max_selected_nodes": max_selected_nodes,
            "max_subquestions": max_subquestions,
        },
        "output_schema": {
            "depth": "int",
            "selected_nodes": [
                {"node_id":"string","why_relevant":"string","priority":"int"}
            ],
            "subquestions": [
                {"sub_id":"string","question":"string","goal":"string","expected_output":"string","related_node_ids":["string"]}
            ],
            "confidence":"float (0~1)",
            "need_more_detail":"bool",
            "notes":"string"
        },
        "json_only": True
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_json_only_instruction() + "\n\n" + to_pretty_json(payload))
    ]


def prompt_generate_code(
    q0: str,
    final_plan: Dict[str, Any],
    used_sources: List[Dict[str, Any]],
) -> List[ChatMessage]:
    system = (
        "你是资深高分子模拟/科学计算工程师。你会根据最终解题思路生成：\n"
        "1) 必要的公式推导（清晰列出变量与假设）；\n"
        "2) 可运行的 Python 代码（默认优先），包含注释、输入参数、输出结果；\n"
        "3) 若你判断 Python 不适合，也可以选择 LAMMPS 或 GROMACS，但必须解释原因，并给出可运行的脚本/输入文件。\n"
        "注意：代码必须与推导一致；必须给出运行方式与预期输出。"
    )
    payload = {
        "task": "CODE_GENERATION",
        "constraints": {
            "must_include_entrypoint_for_python": "run.py",
            "entrypoint_must_be_runnable": True,
            "no_placeholder_code": True
        },
        "question": q0,
        "final_plan": final_plan,
        "used_sources": used_sources,
        "output_schema": {
            "engine_choice": "string (python|lammps|gromacs)",
            "rationale": "string",
            "math_derivation": ["string"],
            "algorithm": ["string"],
            "code_files": [
                {"path":"string","content":"string"}
            ],
            "requirements": ["string (pip package or system dependency)"],
            "run_instructions": ["string"],
            "expected_outputs": ["string"],
            "notes_for_user_to_modify": ["string"]
        },
        "json_only": True
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_json_only_instruction() + "\n\n" + to_pretty_json(payload))
    ]


def prompt_s0_summarize_chunk(book: Dict[str, Any], node: Dict[str, Any], chunk_text: str, chunk_id: str, chunk_index: int, chunk_total: int) -> List[ChatMessage]:
    system = (
        "你是专业的教材梳理助手。你将读取某章节/小节的一个原文分块（chunk），"
        "请提取该 chunk 的关键点、概念和公式，并保持来源 node_id。"
    )
    payload = {
        "task": "S0_NODE_CHUNK_SUMMARY",
        "book": {"book_id": book.get("book_id"), "book_name": book.get("book_name")},
        "node": {"node_id": node.get("node_id"), "level": node.get("level"), "title": node.get("title")},
        "chunk": {"chunk_id": chunk_id, "chunk_index": chunk_index, "chunk_total": chunk_total},
        "output_schema": {
            "node_id": "string",
            "chunk_id": "string",
            "summary": "string",
            "key_points": ["string (<=8)"],
            "key_concepts": ["string (<=12)"],
            "formulas": [{"latex":"string","meaning":"string"}]
        },
        "text": chunk_text,
        "json_only": True
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_json_only_instruction() + "\n\n" + to_pretty_json(payload)),
    ]


def prompt_s0_merge_chunk_summaries(book: Dict[str, Any], node: Dict[str, Any], chunk_summaries: List[Dict[str, Any]]) -> List[ChatMessage]:
    system = (
        "你是专业的教材梳理助手。你将获得同一节点多个 chunk 的摘要信息，请将它们合并为该节点的最终结构化摘要。"
        "合并时去重、保持逻辑结构，并补充必要的承接关系。"
    )
    payload = {
        "task": "S0_NODE_REDUCE",
        "book": {"book_id": book.get("book_id"), "book_name": book.get("book_name")},
        "node": {"node_id": node.get("node_id"), "level": node.get("level"), "title": node.get("title"), "path_titles": node.get("path_titles", [])},
        "chunk_summaries": chunk_summaries,
        "output_schema": {
            "book_id": "string",
            "node_id": "string",
            "level": "int",
            "title": "string",
            "summary": "string",
            "key_points": ["string (<=12)"],
            "key_concepts": ["string (<=20)"],
            "formulas": [{"latex":"string","meaning":"string","variables":[{"symbol":"string","meaning":"string"}]}],
            "cross_links": [{"related_node_id":"string","relation":"string","note":"string"}],
            "source": {"book_name":"string","node_path":"string"}
        },
        "json_only": True
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_json_only_instruction() + "\n\n" + to_pretty_json(payload)),
    ]
