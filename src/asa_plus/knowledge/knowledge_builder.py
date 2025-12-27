from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config_loader import AppConfig, resolve_path, ensure_dirs
from ..llm.router import LLMRouter
from ..llm import prompts as P
from ..utils.logger import get_logger
from ..utils.text_chunker import chunk_text
from .book_parser import BookParser, Node


log = get_logger(__name__)


def _slugify(name: str) -> str:
    import re
    s = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fff]+', '_', name).strip('_')
    return s[:80] if len(s) > 80 else s


class KnowledgeBuilder:
    """
    负责：
    - 解析 books_dir 下的 txt
    - 按章/节/小节切分为 node，并把每个 node 原文保存到 split_books_dir
    - 调用 LLM（API）对指定 level 生成摘要与关键概念，构建 S0
    - 输出：
        - s0_knowledge.json
        - content_index.json
    """

    def __init__(self, config: AppConfig, router: Optional[LLMRouter] = None):
        self.config = config
        ensure_dirs(config)
        self.router = router or LLMRouter(config)

        self.books_dir = resolve_path(config, config.paths.books_dir)
        self.processed_dir = resolve_path(config, config.paths.processed_dir)
        self.split_books_dir = resolve_path(config, config.knowledge.split_books_dir)

        self.s0_file = resolve_path(config, config.knowledge.s0_file)
        self.content_index_file = resolve_path(config, config.knowledge.content_index_file)

        self.summary_levels = set(config.knowledge.summary_levels)
        self.max_chars_per_summary_call = config.knowledge.max_chars_per_summary_call

    def list_books(self) -> List[Path]:
        return sorted([p for p in self.books_dir.glob("*.txt") if p.is_file()])

    def _write_node_text(self, book_id: str, local_id: str, text: str) -> Path:
        book_dir = self.split_books_dir / book_id
        book_dir.mkdir(parents=True, exist_ok=True)
        safe_id = local_id.replace(".", "_")
        path = book_dir / f"{safe_id}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def _summarize_node_once(self, book_meta: Dict[str, Any], node: Node, node_text: str, short: bool = False) -> Dict[str, Any]:
        node_meta = {
            "node_id": node.node_id,
            "level": node.level,
            "title": node.title,
            "path_titles": node.path_titles,
        }
        if short:
            msgs = P.prompt_s0_summarize_node_short(book_meta, node_meta, node_text)
        else:
            msgs = P.prompt_s0_summarize_node(book_meta, node_meta, node_text)
        return self.router.chat_json(task="s0_outline", messages=msgs, model_role="outline")

    def _summarize_node(self, book_meta: Dict[str, Any], node: Node, node_text: str) -> Dict[str, Any]:
        """
        对 node 做摘要：若超长则 map-reduce。
        修复点：失败自动用短 prompt 重试（降低被截断与 JSON 错误概率）。
        """
        if len(node_text) <= self.max_chars_per_summary_call:
            try:
                return self._summarize_node_once(book_meta, node, node_text, short=False)
            except Exception as e:
                log.warning(f"S0 摘要第一次失败，尝试短版重试：{node.node_id} err={e}")
                return self._summarize_node_once(book_meta, node, node_text, short=True)

        # map-reduce
        chunks = chunk_text(
            node_text,
            chunk_size=self.config.agent.chunk_size_chars,
            overlap=self.config.agent.chunk_overlap_chars,
            max_chunks=self.config.agent.max_chunks_per_node,
        )
        chunk_summaries: List[Dict[str, Any]] = []
        node_meta = {
            "node_id": node.node_id,
            "level": node.level,
            "title": node.title,
            "path_titles": node.path_titles,
        }

        for idx, ch in enumerate(chunks, start=1):
            msgs = P.prompt_s0_summarize_chunk(
                book_meta, node_meta,
                chunk_text=ch.text,
                chunk_id=ch.chunk_id,
                chunk_index=idx,
                chunk_total=len(chunks),
            )
            js = self.router.chat_json(task="s0_outline", messages=msgs, model_role="outline")
            chunk_summaries.append(js)

        # reduce
        try:
            msgs = P.prompt_s0_merge_chunk_summaries(book_meta, node_meta, chunk_summaries)
            return self.router.chat_json(task="s0_outline", messages=msgs, model_role="outline")
        except Exception as e:
            # reduce 失败：退化为短版合并（只输出 summary/key_points/key_concepts）
            log.warning(f"S0 reduce 失败，尝试短版合并：{node.node_id} err={e}")
            # 简单合并：把 chunk summaries 的 summary 拼接，再用 short prompt 再压缩一次
            merged_text = "\n\n".join(
                [str(cs.get("summary", "")) for cs in chunk_summaries if isinstance(cs, dict)]
            )
            merged_text = merged_text[: self.max_chars_per_summary_call]
            return self._summarize_node_once(book_meta, node, merged_text, short=True)

    def build_s0(self, force: bool = False) -> Dict[str, Any]:
        if self.s0_file.exists() and self.content_index_file.exists() and not force:
            log.info(f"S0 已存在，跳过构建（force=False）。路径: {self.s0_file}")
            return json.loads(self.s0_file.read_text(encoding="utf-8"))

        books = self.list_books()
        if not books:
            raise FileNotFoundError(f"books_dir 为空：{self.books_dir}，请把 .txt 书籍放进去。")

        global_s0: Dict[str, Any] = {
            "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "books": [],
        }
        content_index: Dict[str, Any] = {}

        for book_path in books:
            book_name = book_path.stem
            book_id = _slugify(book_name)

            log.info(f"解析书籍: {book_name} (book_id={book_id})")
            parser = BookParser(book_path=book_path, book_id=book_id, book_name=book_name)
            clean_text, events = parser.parse()
            nodes = parser.build_nodes(clean_text, events)

            book_meta = {
                "book_id": book_id,
                "book_name": book_name,
                "source_file": str(book_path),
            }

            book_nodes_out: List[Dict[str, Any]] = []

            for node in nodes:
                node_text = parser.extract_node_text(clean_text, node)

                node_file = self._write_node_text(book_id, node.local_id, node_text)

                node_record: Dict[str, Any] = {
                    "book_id": book_id,
                    "book_name": book_name,
                    "node_id": node.node_id,
                    "local_id": node.local_id,
                    "level": node.level,
                    "title": node.title,
                    "parent_id": node.parent_id,
                    "children": node.children,
                    "start_char": node.start_char,
                    "end_char": node.end_char,
                    "path_titles": node.path_titles,
                    "path_str": node.path_str,
                    "text_file": str(node_file),
                }

                if node.level in self.summary_levels:
                    try:
                        summary = self._summarize_node(book_meta, node, node_text)
                        node_record["summary"] = summary
                    except Exception as e:
                        log.warning(f"摘要失败：{node.node_id} {node.title} err={e}")
                        node_record["summary_error"] = str(e)

                book_nodes_out.append(node_record)

                content_index[node.node_id] = {
                    "book_id": book_id,
                    "book_name": book_name,
                    "node_id": node.node_id,
                    "local_id": node.local_id,
                    "level": node.level,
                    "title": node.title,
                    "path_str": node.path_str,
                    "text_file": str(node_file),
                }

            global_s0["books"].append({
                **book_meta,
                "nodes": book_nodes_out
            })

        self.s0_file.parent.mkdir(parents=True, exist_ok=True)
        self.s0_file.write_text(json.dumps(global_s0, ensure_ascii=False, indent=2), encoding="utf-8")
        self.content_index_file.write_text(json.dumps(content_index, ensure_ascii=False, indent=2), encoding="utf-8")

        log.info(f"S0 构建完成: {self.s0_file}")
        log.info(f"content_index 写入: {self.content_index_file}")
        return global_s0
