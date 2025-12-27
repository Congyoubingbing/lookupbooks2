from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    start_char: int
    end_char: int


_LATEX_ENV_BEGIN = re.compile(r"\\begin\{(array|tabular|table|longtable)\}")
_LATEX_ENV_END = re.compile(r"\\end\{(array|tabular|table|longtable)\}")


def _split_into_segments_preserve_env(text: str) -> List[tuple[int, int, str]]:
    """
    把 text 切成若干 segment：
    - 普通文本段：可以合并切块
    - LaTeX 表格环境段：begin..end 作为单段（不可拆）
    返回 [(start,end,seg_text), ...]，start/end 是原文字符位置。
    """
    segments: List[tuple[int, int, str]] = []
    n = len(text)
    i = 0

    # 基于行扫描，能更稳地找到 env
    lines = text.splitlines(True)
    pos = 0
    in_env = False
    env_start = 0
    env_buf: List[str] = []
    normal_buf: List[str] = []
    normal_start = 0

    def flush_normal():
        nonlocal normal_buf, normal_start, pos
        if normal_buf:
            seg_text = "".join(normal_buf)
            seg_end = normal_start + len(seg_text)
            segments.append((normal_start, seg_end, seg_text))
            normal_buf = []

    for ln in lines:
        if not in_env and _LATEX_ENV_BEGIN.search(ln):
            # flush normal first
            flush_normal()
            in_env = True
            env_start = pos
            env_buf = [ln]
        elif in_env:
            env_buf.append(ln)
            if _LATEX_ENV_END.search(ln):
                # flush env
                seg_text = "".join(env_buf)
                seg_end = env_start + len(seg_text)
                segments.append((env_start, seg_end, seg_text))
                env_buf = []
                in_env = False
        else:
            if not normal_buf:
                normal_start = pos
            normal_buf.append(ln)

        pos += len(ln)

    # tail
    if env_buf:
        seg_text = "".join(env_buf)
        seg_end = env_start + len(seg_text)
        segments.append((env_start, seg_end, seg_text))
    else:
        flush_normal()

    return segments


def chunk_text(text: str, chunk_size: int, overlap: int, max_chunks: int) -> List[TextChunk]:
    """
    分块（改进版）：
    - 保证 LaTeX 表格环境 begin..end 不被拆开
    - 其余部分按字符数切块
    - overlap 以字符尾部 overlap 追加到下一块（不跨越 env 段强行拼接）
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须 > 0")
    if overlap < 0:
        raise ValueError("overlap 必须 >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    segments = _split_into_segments_preserve_env(text)
    chunks: List[TextChunk] = []

    buf_text_parts: List[str] = []
    buf_start: int = 0
    buf_end: int = 0

    def flush_buf():
        nonlocal buf_text_parts, buf_start, buf_end
        if not buf_text_parts:
            return
        ctext = "".join(buf_text_parts)
        chunks.append(TextChunk(
            chunk_id=f"chunk_{len(chunks)+1}",
            text=ctext,
            start_char=buf_start,
            end_char=buf_end
        ))
        buf_text_parts = []

    for (seg_start, seg_end, seg_text) in segments:
        if len(chunks) >= max_chunks:
            break

        # 如果当前 buffer 为空，初始化 start
        if not buf_text_parts:
            buf_start = seg_start
            buf_end = seg_start

        # 如果这个 seg 是 env 段（通过 begin 判断），强制单独成块
        is_env = bool(_LATEX_ENV_BEGIN.search(seg_text))

        if is_env:
            # 先 flush 现有 buffer
            flush_buf()
            if len(chunks) >= max_chunks:
                break
            chunks.append(TextChunk(
                chunk_id=f"chunk_{len(chunks)+1}",
                text=seg_text,
                start_char=seg_start,
                end_char=seg_end
            ))
            # env 块之后重新开始
            buf_text_parts = []
            continue

        # 普通段：尝试加入 buffer
        prospective = "".join(buf_text_parts) + seg_text
        if len(prospective) <= chunk_size:
            buf_text_parts.append(seg_text)
            buf_end = seg_end
        else:
            # buffer 先输出
            flush_buf()
            if len(chunks) >= max_chunks:
                break

            # overlap：取上一块尾部 overlap 字符作为新 buffer 开头
            if overlap > 0 and chunks:
                tail = chunks[-1].text[-overlap:]
                # 注意：tail 的 start/end 位置无法精确映射，近似处理为 seg_start-overlap
                buf_text_parts = [tail]
                buf_start = max(0, seg_start - overlap)
                buf_end = buf_start + len(tail)
            else:
                buf_text_parts = []
                buf_start = seg_start
                buf_end = seg_start

            # 再把当前段放入（如果仍超长，就切开这个段）
            if len(seg_text) <= chunk_size:
                buf_text_parts.append(seg_text)
                buf_end = seg_end
            else:
                # 该段本身超长：直接按字符切开（但这是普通段，不是 env）
                s = seg_text
                off = seg_start
                while s and len(chunks) < max_chunks:
                    part = s[:chunk_size]
                    chunks.append(TextChunk(
                        chunk_id=f"chunk_{len(chunks)+1}",
                        text=part,
                        start_char=off,
                        end_char=off + len(part)
                    ))
                    if len(part) >= len(s):
                        s = ""
                        break
                    s = s[chunk_size - overlap:]
                    off = off + (chunk_size - overlap)

                buf_text_parts = []
                continue

    if len(chunks) < max_chunks:
        flush_buf()

    return chunks
