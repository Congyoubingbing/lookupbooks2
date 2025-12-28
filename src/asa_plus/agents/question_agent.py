from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from ..config_loader import AppConfig, resolve_path
from ..knowledge.knowledge_store import KnowledgeStore
from ..llm.router import LLMRouter
from ..llm import prompts as P
from ..utils.logger import get_logger
from ..utils.text_chunker import chunk_text
from .solve_result import SolveResult   # ✅ 新增：稳定返回结构

log = get_logger(__name__)


class QAState(TypedDict, total=False):
    # input
    q0: str
    session_id: str
    depth: int
    s0_outline: str

    # classification history
    classifications: List[Dict[str, Any]]
    current_classification: Dict[str, Any]
    selected_node_ids: List[str]

    # evidence
    chunk_metas: List[Dict[str, Any]]
    evidence_notes: List[Dict[str, Any]]

    # assessment
    assessment: Dict[str, Any]
    final_plan: Dict[str, Any]
    used_sources: List[Dict[str, Any]]

    # control
    max_depth: int
    interactive: bool
    user_confirmed_large_context: bool


class QuestionAgent:
    """
    负责 Q0 -> S1..Sn 的多层分类与推理
    """
    def __init__(self, config: AppConfig, store: KnowledgeStore, router: Optional[LLMRouter] = None):
        self.config = config
        self.store = store
        self.router = router or LLMRouter(config)

        self.runtime_dir = resolve_path(config, config.execution.local.workdir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def solve(self, question: str) -> SolveResult:
        """
        处理 Q0 问题的推理过程
        """
        session_id = uuid.uuid4().hex[:8]
        outline = self.store.render_outline(max_level=2)

        init_state: QAState = {
            "q0": question,
            "session_id": session_id,
            "depth": 1,
            "s0_outline": outline,
            "classifications": [],
            "max_depth": self.config.agent.max_depth,
            "interactive": True,
            "user_confirmed_large_context": False,
        }

        state = self._process_question(init_state)

        result: Dict[str, Any] = {
            "session_id": state["session_id"],
            "final_state": state["final_state"],
            "final_plan": state["final_plan"],
            "plan_text": state.get("plan_text", "解题思路未生成"),
            "assessment": state["assessment"],
            "used_sources": state["used_sources"],
        }

        log.info(f"当前推理状态: {state}")

        # ✅ 唯一升级点：稳定封装（不改变任何字段）
        return SolveResult.from_dict(result)

    def _process_question(self, init_state: QAState) -> Dict[str, Any]:
        app = self.build_graph()
        final_state = app.invoke(init_state)

        assessment = final_state.get("assessment", {})
        final_plan = {
            "depth": final_state.get("depth", 1),
            "classifications": final_state.get("classifications", []),
            "assessment": assessment,
        }

        plan_text = self._generate_plan_text(final_state)

        return {
            "session_id": init_state["session_id"],
            "final_state": final_state,
            "final_plan": final_plan,
            "assessment": assessment,
            "used_sources": final_state.get("used_sources", []),
            "plan_text": plan_text,
        }

    def _generate_plan_text(self, final_state: Dict[str, Any]) -> str:
        classifications = final_state.get("classifications", [])
        if not classifications:
            log.warning("解题思路生成失败：分类结果为空")
            return "解题思路未生成"

        plan_lines = ["解题思路："]
        for classification in classifications:
            plan_lines.append(
                f"- {classification.get('node_id', '未知节点')} 选中章节：{classification.get('title', '无标题')}"
            )
        return "\n".join(plan_lines)

    # ---------------------------
    # LangGraph nodes（以下全部保持不变）
    # ---------------------------

    def _node_make_classification(self, state: QAState) -> Dict[str, Any]:
        depth = state.get("depth", 1)
        q0 = state["q0"]
        s0_outline = state["s0_outline"]
        classifications = state.get("classifications", [])

        if depth == 1 and not classifications:
            msgs = P.prompt_decompose_question(
                q0=q0,
                s0_outline=s0_outline,
                depth=depth,
                max_selected_nodes=self.config.agent.max_selected_nodes,
                max_subquestions=self.config.agent.max_subquestions,
                prev_classification=None,
            )
            S = self.router.chat_json(task="decomposition", messages=msgs, model_role="reasoning")
        else:
            last_S = classifications[-1]
            last_assess = state.get("assessment", {})
            msgs = P.prompt_refine_classification(
                q0=q0,
                depth_next=depth,
                s0_outline=s0_outline,
                last_classification=last_S,
                last_assessment=last_assess,
                max_selected_nodes=self.config.agent.max_selected_nodes,
                max_subquestions=self.config.agent.max_subquestions,
            )
            S = self.router.chat_json(task="decomposition", messages=msgs, model_role="reasoning")

        selected = [x.get("node_id", "") for x in S.get("selected_nodes", []) if isinstance(x, dict)]
        selected = self.store.normalize_node_ids(selected)[: self.config.agent.max_selected_nodes]

        return {
            "current_classification": S,
            "classifications": classifications + [S],
            "selected_node_ids": selected,
        }

    def _node_retrieve_and_chunk(self, state: QAState) -> Dict[str, Any]:
        session_id = state["session_id"]
        depth = state["depth"]
        selected = state.get("selected_node_ids", [])

        chunk_metas: List[Dict[str, Any]] = []
        total_chunks = 0

        chunk_root = self.runtime_dir / "chunks" / session_id / f"depth_{depth}"
        if chunk_root.exists():
            for p in chunk_root.glob("**/*"):
                if p.is_file():
                    p.unlink()
        chunk_root.mkdir(parents=True, exist_ok=True)

        for node_id in selected:
            rec = self.store.get_node_record(node_id)
            node_text = self.store.get_node_text(node_id)

            chunks = chunk_text(
                node_text,
                chunk_size=self.config.agent.chunk_size_chars,
                overlap=self.config.agent.chunk_overlap_chars,
                max_chunks=self.config.agent.max_chunks_per_node,
            )
            total_chunks += len(chunks)

            node_dir = chunk_root / node_id.replace("::", "__").replace("/", "_")
            node_dir.mkdir(parents=True, exist_ok=True)

            for i, ch in enumerate(chunks, start=1):
                ch_file = node_dir / f"{ch.chunk_id}.txt"
                ch_file.write_text(ch.text, encoding="utf-8")

                chunk_metas.append({
                    "book_id": rec.get("book_id"),
                    "book_name": rec.get("book_name"),
                    "node_id": node_id,
                    "title": rec.get("title"),
                    "path_str": rec.get("path_str"),
                    "chunk_id": ch.chunk_id,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "chunk_file": str(ch_file),
                })

        log.info(f"[depth={depth}] 原文切分完成，总 chunk={total_chunks}")
        return {"chunk_metas": chunk_metas, "evidence_notes": []}

    def _node_extract_evidence(self, state: QAState) -> Dict[str, Any]:
        q0 = state["q0"]
        depth = state["depth"]
        chunk_metas = state.get("chunk_metas", [])
        notes: List[Dict[str, Any]] = []

        for meta in chunk_metas:
            chunk_text = Path(meta["chunk_file"]).read_text(encoding="utf-8", errors="ignore")
            msgs = P.prompt_extract_evidence_from_chunk(
                q0=q0,
                depth=depth,
                node_meta=meta,
                chunk_text=chunk_text,
                chunk_id=meta["chunk_id"],
                chunk_index=meta["chunk_index"],
                chunk_total=meta["chunk_total"],
            )
            js = self.router.chat_json(task="content_evidence", messages=msgs, model_role="reasoning")
            notes.append(js)

        return {"evidence_notes": notes}

    def _node_integrate_and_assess(self, state: QAState) -> Dict[str, Any]:
        msgs = P.prompt_integrate_evidence(
            q0=state["q0"],
            depth=state["depth"],
            classification=state["current_classification"],
            s0_outline=state["s0_outline"],
            evidence_notes=state.get("evidence_notes", []),
        )
        assessment = self.router.chat_json(task="integration", messages=msgs, model_role="reasoning")
        return {"assessment": assessment, "used_sources": assessment.get("used_sources", [])}

    def _node_prepare_next_depth(self, state: QAState) -> Dict[str, Any]:
        return {
            "depth": state["depth"] + 1,
            "current_classification": {},
            "selected_node_ids": [],
            "chunk_metas": [],
            "evidence_notes": [],
        }

    def _should_continue(self, state: QAState) -> str:
        if state.get("assessment", {}).get("can_solve"):
            return "finish"
        if state["depth"] >= state["max_depth"]:
            return "finish"
        return "refine"

    def build_graph(self):
        g = StateGraph(QAState)

        g.add_node("make_classification", self._node_make_classification)
        g.add_node("retrieve_and_chunk", self._node_retrieve_and_chunk)
        g.add_node("extract_evidence", self._node_extract_evidence)
        g.add_node("integrate_and_assess", self._node_integrate_and_assess)
        g.add_node("prepare_next", self._node_prepare_next_depth)

        g.set_entry_point("make_classification")
        g.add_edge("make_classification", "retrieve_and_chunk")
        g.add_edge("retrieve_and_chunk", "extract_evidence")
        g.add_edge("extract_evidence", "integrate_and_assess")

        g.add_conditional_edges(
            "integrate_and_assess",
            self._should_continue,
            {"finish": END, "refine": "prepare_next"},
        )
        g.add_edge("prepare_next", "make_classification")

        return g.compile()
