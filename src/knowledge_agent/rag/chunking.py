"""分句 / 分块，生成 solid_sentences.json / solid_chunks.json。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ..paths import CHUNKS_FILE, SENTENCES_FILE, resolve_project_path

CHUNK_SIZE = 300
MAX_SIZE_CAPABLE = 512

RE_VERSION_DATE = re.compile(r"^版本日期\s*[:：]\s*(\d{4}-\d{2}-\d{2})")


def _split_long_sentence(sentence: str) -> list[str]:
    """把超长句按上限切成若干近似等长的片段。"""
    pieces = math.ceil(len(sentence) / MAX_SIZE_CAPABLE)
    size = math.ceil(len(sentence) / pieces)
    return [sentence[i : i + size] for i in range(0, len(sentence), size)]


def get_sentences(raw_text: str) -> list[str]:
    """按中文标点分句；单句超过 512 字时再切分。"""
    parts = re.split(r"(?<=[，。！？；\n])", raw_text)
    sentences = [part.strip() for part in parts if part.strip()]
    result: list[str] = []
    for sentence in sentences:
        if len(sentence) > MAX_SIZE_CAPABLE:
            result.extend(_split_long_sentence(sentence))
        else:
            result.append(sentence)
    return result


def chunk_up(sentences: list[str]) -> list[str]:
    """把句子合并成约 300 字的小段落，相邻块之间保留最后一句作为重叠。"""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for sentence in sentences:
        if not sentence:
            continue
        if buffer and buffer_len + len(sentence) > CHUNK_SIZE:
            chunks.append("".join(buffer))
            # 重叠一句，但不要超过单块上限。
            last = buffer[-1]
            if len(last) + len(sentence) <= MAX_SIZE_CAPABLE:
                buffer = [last]
                buffer_len = len(last)
            else:
                buffer = []
                buffer_len = 0
        buffer.append(sentence)
        buffer_len += len(sentence)

    if buffer:
        chunks.append("".join(buffer))
    return chunks


def _strip_version_line(raw_text: str) -> str:
    """去掉首行的"版本日期"元信息，避免它被当成正文检索。"""
    lines = raw_text.splitlines()
    if not lines:
        return raw_text
    first_line = lines[0].strip().lstrip("\ufeff")
    if RE_VERSION_DATE.match(first_line):
        return "\n".join(lines[1:])
    return raw_text


def split_it_up(
    sorted_pipeline_file_list: list[dict],
) -> tuple[list[dict], list[dict]]:
    """把文档列表切成 solid_sentences / solid_chunks 并写盘。"""
    solid_sentences: list[dict] = []
    solid_chunks: list[dict] = []

    for file_info in sorted_pipeline_file_list:
        source = resolve_project_path(file_info["path"])
        raw_text = source.read_text(encoding="utf-8")
        if not raw_text.strip():
            continue
        raw_text = _strip_version_line(raw_text)

        sentences = get_sentences(raw_text)
        chunks = chunk_up(sentences)
        solid_sentences.extend([{**file_info, "content": sentence} for sentence in sentences])
        solid_chunks.extend([{**file_info, "content": chunk} for chunk in chunks])

    SENTENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SENTENCES_FILE.write_text(
        json.dumps(solid_sentences, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    CHUNKS_FILE.write_text(
        json.dumps(solid_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return solid_sentences, solid_chunks
