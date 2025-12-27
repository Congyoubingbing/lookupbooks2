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


log = get_logger(__name__)


class QAState(TypedDict, total=False):
    # input
    q0: str
    session_id: str
    depth: int
    s0_outline: str

    # classification history
    classifications: List[Dict[str, Any]]   # S1..Sk
    current_classification: Dict[str, Any]
    selected_node_ids: List[str]

    # evidence
    chunk_metas: List[Dict[str, Any]]
    evidence_notes: List[Dict[str, Any]]

    # assessment
    assessment: Dict[str, Any]              # integrate result
    final_plan: Dict[str, Any]
    used_sources: List[Dict[str, Any]]

    # control
    max_depth: int
    interactive: bool
    user_confirmed_large_context: bool


class QuestionAgent:
    """
    负责 Q0 -> S1..Sn 的多层分类与推理（严格遵循你的步骤描述）：
    ① Q0 + S0 -> S1（分类：选择相关章节/子类）
    ② S1 + Q0 + S0 -> 基于选中章节“全部原文”推理，判断能否解决；不能则再分类生成 S2
    ③ 重复直到 Sn 可解决或达到 max_depth

    关键：对每轮选中章节，系统会把该章节原文切成多个 chunk 并逐块发送给 LLM（保证“全部内容”被使用）。
    """
    def __init__(self, config: AppConfig, store: KnowledgeStore, router: Optional[LLMRouter] = None):
        self.config = config
        self.store = store
        self.router = router or LLMRouter(config)

        self.runtime_dir = resolve_path(config, config.execution.local.workdir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def solve(self, question: str) -> Dict[str, Any]:
        """
        处理 Q0 问题的推理过程：
        1. 分类（Q0 + S0 -> S1..Sn）
        2. 基于选中章节推理，判断是否可以解决
        3. 如果不能解决，则进一步细化分类生成 S2..Sn，直到解决问题
        """
        # 初始化推理状态
        session_id = uuid.uuid4().hex[:8]
        outline = self.store.render_outline(max_level=2)  # 获取 S0 outline

        # 初始状态
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

        # 推理过程：多层分类、推理、证据提取
        state = self._process_question(init_state)

        # 确保返回结果包含 'plan_text'
        result = {
            "session_id": state["session_id"],
            "final_state": state["final_state"],
            "final_plan": state["final_plan"],
            "plan_text": state.get("plan_text", "解题思路未生成"),  # 添加 plan_text 字段
            "assessment": state["assessment"],
            "used_sources": state["used_sources"]
        }
        log.info(f"当前推理状态: {state}")
        return result

    def _process_question(self, init_state: QAState) -> Dict[str, Any]:
        """
        处理问题推理过程，包括分类、证据提取、推理评估等步骤
        """
        app = self.build_graph()  # 创建 LangGraph 状态图
        final_state = app.invoke(init_state)  # 启动推理过程

        # 处理最终推理结果
        assessment = final_state.get("assessment", {})
        final_plan = {
            "depth": final_state.get("depth", 1),
            "classifications": final_state.get("classifications", []),
            "assessment": assessment,
        }

        # 在结果中生成解题思路
        plan_text = self._generate_plan_text(final_state)


        # 返回推理结果、评估、解题思路
        return {
            "session_id": init_state["session_id"],
            "final_state": final_state,
            "final_plan": final_plan,
            "assessment": assessment,
            "used_sources": final_state.get("used_sources", []),
            "plan_text": plan_text  # 返回解题思路
        }

    def _generate_plan_text(self, final_state: Dict[str, Any]) -> str:
        classifications = final_state.get("classifications", [])
        if not classifications:
            log.warning("解题思路生成失败：分类结果为空！请检查分类逻辑。")
            return "解题思路未生成"

        plan_lines = ["解题思路："]
        for classification in classifications:
            plan_lines.append(
                f"- {classification.get('node_id', '未知节点')} 选中章节：{classification.get('title', '无标题')}")

        plan_text = "\n".join(plan_lines)
        return plan_text

    # ---------------------------
    # LangGraph 节点函数
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

        # 提取 selected_node_ids 并校验
        selected = [x.get("node_id","") for x in S.get("selected_nodes", []) if isinstance(x, dict)]
        selected = self.store.normalize_node_ids(selected)[: self.config.agent.max_selected_nodes]

        new_hist = classifications + [S]
        log.info(f"[depth={depth}] 选中节点数: {len(selected)}")

        return {
            "current_classification": S,
            "classifications": new_hist,
            "selected_node_ids": selected,
        }

    def _node_retrieve_and_chunk(self, state: QAState) -> Dict[str, Any]:
        """
        读取选中章节/小节原文，并切 chunk，保存到运行目录。
        """
        session_id = state["session_id"]
        depth = state["depth"]
        selected = state.get("selected_node_ids", [])

        chunk_metas: List[Dict[str, Any]] = []
        total_chunks = 0

        chunk_root = self.runtime_dir / "chunks" / session_id / f"depth_{depth}"
        if chunk_root.exists():
            # 清理旧的
            for p in chunk_root.glob("**/*"):
                if p.is_file():
                    p.unlink()
        chunk_root.mkdir(parents=True, exist_ok=True)

        for node_id in selected:
            rec = self.store.get_node_record(node_id)
            idx = self.store.get_node_index(node_id)
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

        # 大上下文成本提示（可选）
        if state.get("interactive", True) and not state.get("user_confirmed_large_context", False):
            if total_chunks >= self.config.agent.require_user_confirm_if_total_chunks_ge:
                print("\n[警告] 本轮将发送给模型的原文 chunk 数量较大：", total_chunks)
                print("这意味着调用次数与费用可能较高。")
                ans = input("是否继续本轮推理？(y/n): ").strip().lower()
                if ans != "y":
                    raise SystemExit("用户取消（原文 chunk 过多）")
                state["user_confirmed_large_context"] = True

        log.info(f"[depth={depth}] 原文切分完成，总 chunk={total_chunks}")
        return {"chunk_metas": chunk_metas, "evidence_notes": []}

    def _node_extract_evidence(self, state: QAState) -> Dict[str, Any]:
        """
        对每个 chunk 调用 LLM 提取证据 notes。
        """
        q0 = state["q0"]
        depth = state["depth"]
        chunk_metas = state.get("chunk_metas", [])
        notes: List[Dict[str, Any]] = []

        for k, meta in enumerate(chunk_metas, start=1):
            chunk_text = Path(meta["chunk_file"]).read_text(encoding="utf-8", errors="ignore")
            node_meta = {
                "book_id": meta.get("book_id"),
                "book_name": meta.get("book_name"),
                "node_id": meta.get("node_id"),
                "title": meta.get("title"),
                "path_str": meta.get("path_str"),
            }
            msgs = P.prompt_extract_evidence_from_chunk(
                q0=q0,
                depth=depth,
                node_meta=node_meta,
                chunk_text=chunk_text,
                chunk_id=meta["chunk_id"],
                chunk_index=meta["chunk_index"],
                chunk_total=meta["chunk_total"],
            )
            js = self.router.chat_json(task="content_evidence", messages=msgs, model_role="reasoning")
            notes.append(js)

            if k % 10 == 0:
                log.info(f"[depth={depth}] evidence progress: {k}/{len(chunk_metas)}")

        return {"evidence_notes": notes}

    def _node_integrate_and_assess(self, state: QAState) -> Dict[str, Any]:
        q0 = state["q0"]
        depth = state["depth"]
        S = state["current_classification"]
        s0_outline = state["s0_outline"]
        evidence = state.get("evidence_notes", [])

        msgs = P.prompt_integrate_evidence(
            q0=q0,
            depth=depth,
            classification=S,
            s0_outline=s0_outline,
            evidence_notes=evidence,
        )
        assessment = self.router.chat_json(task="integration", messages=msgs, model_role="reasoning")

        used_sources = assessment.get("used_sources", [])
        return {"assessment": assessment, "used_sources": used_sources}

    def _node_prepare_next_depth(self, state: QAState) -> Dict[str, Any]:
        depth = state["depth"]
        return {
            "depth": depth + 1,
            "current_classification": {},
            "selected_node_ids": [],
            "chunk_metas": [],
            "evidence_notes": [],
        }

    def _should_continue(self, state: QAState) -> str:
        depth = state["depth"]
        max_depth = state["max_depth"]
        assessment = state.get("assessment", {})
        can = bool(assessment.get("can_solve", False))
        conf = float(assessment.get("confidence", 0.0) or 0.0)

        if can:
            return "finish"
        if depth >= max_depth:
            return "finish"
        # 允许在置信度高时提前停止细分（但仍会进入 finish）
        if conf >= self.config.agent.stop_if_confidence_ge and assessment.get("solution_outline"):
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

        g.add_conditional_edges("integrate_and_assess", self._should_continue, {
            "finish": END,
            "refine": "prepare_next",
        })
        g.add_edge("prepare_next", "make_classification")

        return g.compile()
