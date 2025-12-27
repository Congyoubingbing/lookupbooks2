from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config_loader import AppConfig, resolve_path
from ..utils.logger import get_logger


log = get_logger(__name__)


class KnowledgeStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.s0_path = resolve_path(config, config.knowledge.s0_file)
        self.index_path = resolve_path(config, config.knowledge.content_index_file)

        # 检查 S0 文件和索引文件是否存在
        if not self.s0_path.exists() or not self.index_path.exists():
            raise FileNotFoundError(
                f"找不到 S0 或索引文件。请先运行 build-s0。\nS0: {self.s0_path}\nIndex: {self.index_path}"
            )

        self.s0: Dict[str, Any] = {}  # 初始化 s0 数据
        self.index: Dict[str, Any] = {}  # 初始化索引

        # 加载数据
        self.load()

        # 构建 node_id -> node_record 的映射（包含 summary 等）
        self.node_records: Dict[str, Dict[str, Any]] = {}
        for b in self.s0.get("books", []):
            for n in b.get("nodes", []):
                self.node_records[n["node_id"]] = n

    def load(self):
        """
        加载 S0 知识框架文件（s0_knowledge.json）和索引文件（content_index.json）。
        """
        try:
            # 加载 S0 文件
            with open(self.s0_path, "r", encoding="utf-8") as f:
                self.s0 = json.load(f)
            log.info(f"S0 数据加载成功: {self.s0_path}")

            # 加载索引文件
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
            log.info(f"内容索引加载成功: {self.index_path}")

        except Exception as e:
            log.error(f"加载 S0 或索引文件失败: {e}")
            raise

    def has_node(self, node_id: str) -> bool:
        return node_id in self.node_records

    def get_node_record(self, node_id: str) -> Dict[str, Any]:
        if node_id not in self.node_records:
            raise KeyError(f"未知 node_id: {node_id}")
        return self.node_records[node_id]

    def get_node_index(self, node_id: str) -> Dict[str, Any]:
        if node_id not in self.index:
            raise KeyError(f"索引中未知 node_id: {node_id}")
        return self.index[node_id]

    def get_node_text(self, node_id: str) -> str:
        info = self.get_node_index(node_id)
        p = Path(info["text_file"])
        if not p.exists():
            raise FileNotFoundError(f"找不到节点原文文件: {p}")
        return p.read_text(encoding="utf-8", errors="ignore")

    def render_outline(self, max_level: int = 2) -> str:
        """
        输出给 LLM 的 S0 outline：
        - 仅包含 node_id + title
        - 带缩进体现层级
        """
        lines: List[str] = []
        for b in self.s0.get("books", []):
            lines.append(f"=== BOOK: {b.get('book_name')} (book_id={b.get('book_id')}) ===")
            # 按出现顺序输出
            nodes = b.get("nodes", [])
            # 先构建 parent -> children
            children_map: Dict[Optional[str], List[Dict[str, Any]]] = {}
            for n in nodes:
                if n.get("level", 1) > max_level:
                    continue
                parent = n.get("parent_id")
                children_map.setdefault(parent, []).append(n)

            def walk(parent_id: Optional[str], indent: int):
                for n in children_map.get(parent_id, []):
                    node_id = n["node_id"]
                    title = n["title"]
                    lvl = n["level"]
                    lines.append(f"{'  '*indent}[{node_id}] {title}")
                    walk(node_id, indent + 1)

            walk(None, 0)
            lines.append("")  # blank line
        return "\n".join(lines).strip()

    def render_outline_subset(self, node_ids: List[str], include_children: bool = True, max_level: int = 3) -> str:
        """
        针对选中节点输出更小的 outline（用于 refine），可选包含子节点。
        """
        lines: List[str] = []
        seen = set()

        def add_node(nid: str, indent: int):
            if nid in seen:
                return
            seen.add(nid)
            rec = self.get_node_record(nid)
            if rec.get("level", 1) > max_level:
                return
            lines.append(f"{'  '*indent}[{nid}] {rec.get('title')}")
            if include_children:
                for cid in rec.get("children", []):
                    if self.has_node(cid):
                        add_node(cid, indent + 1)

        for nid in node_ids:
            if self.has_node(nid):
                add_node(nid, 0)

        return "\n".join(lines).strip()

    def normalize_node_ids(self, node_ids: List[str]) -> List[str]:
        """过滤掉不存在的 id，并去重保持顺序。"""
        out: List[str] = []
        seen = set()
        for nid in node_ids:
            if nid in self.node_records and nid not in seen:
                seen.add(nid)
                out.append(nid)
        return out
