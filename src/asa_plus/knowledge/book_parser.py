from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..utils.logger import get_logger


log = get_logger(__name__)


@dataclass
class HeadingEvent:
    level: int
    title: str
    line_no: int
    char_pos: int
    raw_line: str


@dataclass
class Node:
    book_id: str
    book_name: str
    local_id: str
    node_id: str
    level: int
    title: str
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0
    path_titles: List[str] = field(default_factory=list)

    @property
    def path_str(self) -> str:
        return " > ".join(self.path_titles) if self.path_titles else self.title


class BookParser:
    """
    将书籍 txt 解析为章/节/小节节点。

    关键修复：
    - LaTeX 表格环境（array/tabular/table/longtable）内禁止标题识别，避免把表格行当标题。
    - 对表格行（含 & 或 \\ 等）增加强过滤，不参与 numbered heading 识别。
    """

    # LaTeX headings
    _re_chapter = re.compile(r'^\s*\\chapter\*?\{(.+?)\}\s*$')
    _re_section = re.compile(r'^\s*\\section\*?\{(.+?)\}\s*$')
    _re_subsection = re.compile(r'^\s*\\subsection\*?\{(.+?)\}\s*$')
    _re_subsubsection = re.compile(r'^\s*\\subsubsection\*?\{(.+?)\}\s*$')

    # Plain headings (Chinese)
    _re_cn_chapter = re.compile(r'^\s*第\s*([0-9一二三四五六七八九十百千万]+)\s*章\s*(.*)\s*$')
    # Numbered headings: 1.2 / 1.2.3 etc
    _re_num_heading = re.compile(r'^\s*(\d+(?:\.\d+){1,3})\s+(.+?)\s*$')
    _re_chapter_word = re.compile(r'^\s*Chapter\s+(\d+)\s*[:.\-]?\s*(.*)\s*$', re.IGNORECASE)

    _re_toc_marker = re.compile(r'\\section\*?\{Contents\}|\bContents\b|目录|目\s*录', re.IGNORECASE)
    _re_page_sep = re.compile(r'^\s*---\s*$')
    _re_ignore_heading_line = re.compile(r'\\hfill|\\dotfill|\.{3,}|\s\d+\s*$')

    # LaTeX env guard
    _re_latex_env_begin = re.compile(r'\\begin\{(array|tabular|table|longtable)\}')
    _re_latex_env_end = re.compile(r'\\end\{(array|tabular|table|longtable)\}')

    def __init__(self, book_path: Path, book_id: str, book_name: Optional[str] = None):
        self.book_path = book_path
        self.book_id = book_id
        self.book_name = book_name or book_path.stem

    @staticmethod
    def _slugify(name: str) -> str:
        s = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fff]+', '_', name).strip('_')
        return s[:80] if len(s) > 80 else s

    def read_text(self) -> str:
        return self.book_path.read_text(encoding="utf-8", errors="ignore")

    def _strip_preamble(self, text: str) -> str:
        idx = text.find("\\begin{document}")
        if idx >= 0:
            return text[idx + len("\\begin{document}"):]
        return text

    def _strip_toc(self, text: str) -> str:
        """
        粗略移除目录块：从“Contents/目录”标记开始，直到遇到 '---' 分页分隔。
        """
        lines = text.splitlines(True)
        out_lines: List[str] = []
        in_toc = False
        for line in lines:
            if not in_toc and self._re_toc_marker.search(line):
                in_toc = True
                continue
            if in_toc:
                if self._re_page_sep.match(line):
                    in_toc = False
                continue
            out_lines.append(line)
        return "".join(out_lines)

    @staticmethod
    def _looks_like_table_row(line: str) -> bool:
        """
        强过滤：表格行不要当标题。
        常见：包含 '&'，并可能包含 '\\\\' 或 '\\times'。
        """
        s = line.strip()
        if "&" in s:
            return True
        if s.endswith(r"\\"):
            return True
        if r"\times" in s:
            return True
        # array/tabular 的列定义行：{|c|c|...}
        if re.match(r'^\s*\{\s*\|?[lcr]\s*\|?', s):
            return True
        return False

    def _detect_heading(self, line: str) -> Optional[Tuple[int, str]]:
        """
        返回 (level, title) 或 None。
        level：1=章，2=节，3=小节
        """
        if not line:
            return None

        # 目录页码/点线/\\hfill 行过滤
        if self._re_ignore_heading_line.search(line):
            return None

        # 表格行强过滤（避免 0.0133 & ... 被当作 0.0133 标题）
        if self._looks_like_table_row(line):
            return None

        m = self._re_chapter.match(line)
        if m:
            return 1, m.group(1).strip()

        m = self._re_section.match(line)
        if m:
            title = m.group(1).strip()
            if re.match(r'^\d+\.\d+', title):
                return 2, title
            return 1, title

        m = self._re_subsection.match(line)
        if m:
            return 2, m.group(1).strip()

        m = self._re_subsubsection.match(line)
        if m:
            return 3, m.group(1).strip()

        m = self._re_cn_chapter.match(line)
        if m:
            num = m.group(1).strip()
            tail = m.group(2).strip()
            title = f"第{num}章 {tail}".strip()
            return 1, title

        m = self._re_chapter_word.match(line)
        if m:
            num = m.group(1).strip()
            tail = m.group(2).strip()
            title = f"Chapter {num} {tail}".strip()
            return 1, title

        m = self._re_num_heading.match(line)
        if m:
            label = m.group(1).strip()
            tail = m.group(2).strip()
            # 防止把 0.0133 & 误判：上面已过滤 &，这里仍再保守一次
            if "&" in tail or tail.endswith(r"\\"):
                return None
            level = 2 if label.count(".") == 1 else 3
            title = f"{label} {tail}".strip()
            return level, title

        return None

    def parse(self) -> Tuple[str, List[HeadingEvent]]:
        """
        返回 (clean_text, heading_events)
        clean_text: 去掉 preamble/TOC 后的文本
        heading_events: 识别到的标题事件列表

        修复点：LaTeX 表格环境内不做标题检测。
        """
        raw = self.read_text()
        text = self._strip_preamble(raw)
        text = self._strip_toc(text)

        lines = text.splitlines(True)
        events: List[HeadingEvent] = []
        char_pos = 0

        in_env = False
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 进入/退出表格环境
            if not in_env and self._re_latex_env_begin.search(stripped):
                in_env = True
            if in_env:
                # 环境内完全跳过标题检测
                if self._re_latex_env_end.search(stripped):
                    in_env = False
                char_pos += len(line)
                continue

            det = self._detect_heading(stripped)
            if det:
                level, title = det
                events.append(HeadingEvent(
                    level=level,
                    title=title,
                    line_no=i,
                    char_pos=char_pos,
                    raw_line=stripped
                ))
            char_pos += len(line)

        return text, events

    @staticmethod
    def _extract_numeric_label(title: str) -> Optional[str]:
        m = re.match(r'^\s*第\s*([0-9]+)\s*章', title)
        if m:
            return m.group(1)

        m = re.match(r'^\s*Chapter\s+([0-9]+)\b', title, re.IGNORECASE)
        if m:
            return m.group(1)

        m = re.match(r'^\s*(\d+(?:\.\d+){0,3})\b', title)
        if m:
            return m.group(1)

        return None

    def build_nodes(self, text: str, events: List[HeadingEvent]) -> List[Node]:
        if not events:
            n = Node(
                book_id=self.book_id,
                book_name=self.book_name,
                local_id="1",
                node_id=f"{self.book_id}::1",
                level=1,
                title="全文",
                parent_id=None,
                start_char=0,
                end_char=len(text),
                path_titles=[self.book_name, "全文"]
            )
            return [n]

        temp_nodes: List[Node] = []
        for idx, ev in enumerate(events, start=1):
            temp_nodes.append(Node(
                book_id=self.book_id,
                book_name=self.book_name,
                local_id=f"tmp_{idx}",
                node_id=f"{self.book_id}::tmp_{idx}",
                level=ev.level,
                title=ev.title,
                parent_id=None,
                start_char=ev.char_pos,
                end_char=len(text),
                path_titles=[],
            ))

        stack: List[int] = []
        for i, node in enumerate(temp_nodes):
            while stack and temp_nodes[stack[-1]].level >= node.level:
                stack.pop()
            if stack:
                parent = temp_nodes[stack[-1]]
                node.parent_id = parent.node_id
                parent.children.append(node.node_id)
            stack.append(i)

        for i, node in enumerate(temp_nodes):
            end = len(text)
            for j in range(i + 1, len(temp_nodes)):
                if temp_nodes[j].level <= node.level:
                    end = temp_nodes[j].start_char
                    break
            node.end_char = end

        chapters = [n for n in temp_nodes if n.level == 1]
        next_ch_num = 1
        for ch in chapters:
            label = self._extract_numeric_label(ch.title)
            if label and label.isdigit():
                ch_local = label
            else:
                ch_local = str(next_ch_num)
                next_ch_num += 1
            ch.local_id = ch_local

        children_by_parent: Dict[str, List[Node]] = {}
        for n in temp_nodes:
            if n.parent_id:
                children_by_parent.setdefault(n.parent_id, []).append(n)

        def assign_children(parent: Node, parent_local: str):
            kids = children_by_parent.get(parent.node_id, [])
            seq = 1
            for k in kids:
                label = self._extract_numeric_label(k.title)
                if label and label.startswith(parent_local):
                    k_local = label
                else:
                    k_local = f"{parent_local}.{seq}"
                    seq += 1
                k.local_id = k_local
                assign_children(k, k_local)

        roots = [n for n in temp_nodes if n.parent_id is None]
        for r in roots:
            if r.level == 1:
                assign_children(r, r.local_id)
            else:
                label = self._extract_numeric_label(r.title)
                if not label:
                    r.local_id = f"0.{roots.index(r) + 1}"

        old_to_new: Dict[str, str] = {}
        for n in temp_nodes:
            old_to_new[n.node_id] = f"{self.book_id}::{n.local_id}"
        for n in temp_nodes:
            n.node_id = old_to_new[n.node_id]
        for n in temp_nodes:
            if n.parent_id:
                n.parent_id = old_to_new.get(n.parent_id, n.parent_id)
            n.children = [old_to_new.get(cid, cid) for cid in n.children]

        node_by_id = {n.node_id: n for n in temp_nodes}

        def build_path(n: Node) -> List[str]:
            parts = [self.book_name]
            chain = []
            cur: Optional[Node] = n
            while cur is not None:
                chain.append(cur.title)
                if cur.parent_id:
                    cur = node_by_id.get(cur.parent_id)
                else:
                    cur = None
            parts.extend(reversed(chain))
            return parts

        for n in temp_nodes:
            n.path_titles = build_path(n)

        return temp_nodes

    def extract_node_text(self, text: str, node: Node) -> str:
        return text[node.start_char:node.end_char].strip()
