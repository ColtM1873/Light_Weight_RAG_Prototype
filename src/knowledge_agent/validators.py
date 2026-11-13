"""Guard 层的纯校验 / 构造函数（不依赖 LLM，可单测）。

- missing_required_fields: 结构化结果是否缺必填内容
- validate_citations: 引用是否与 solid_sentences / solid_chunks 对得上
- make_feedback: 把"缺什么 / 错在哪"变成委婉的人话 HumanMessage
"""

from __future__ import annotations

import difflib
import json
import re
import uuid
from functools import lru_cache
from typing import Any, Sequence

from langchain_core.messages import HumanMessage

from .paths import CHUNKS_FILE, SENTENCES_FILE
from .prompts import (
    FEEDBACK_BAD_REFS,
    FEEDBACK_FINALIZE,
    FEEDBACK_MISSING_FIELDS,
    FEEDBACK_UNSUPPORTED,
)
from .schemas import StructuredAnswer

# 反馈里最多列几条，避免 HumanMessage 过长
MAX_FEEDBACK_ITEMS = 10


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_solid_library() -> tuple[list[dict], list[dict]] | None:
    """懒加载 solid 数据；文件不存在或损坏时返回 None（调用方跳过引用校验）。"""
    try:
        sentences = json.loads(SENTENCES_FILE.read_text(encoding="utf-8"))
        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(sentences, list) or not isinstance(chunks, list):
        return None
    return sentences, chunks


def clear_library_cache() -> None:
    """索引重建后清掉引用校验用的缓存。"""
    load_solid_library.cache_clear()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def content_to_text(content: Any) -> str:
    """把消息 content（str / list[dict] / 其它）转成可读文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(text if isinstance(text, str) else str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _normalize(text: str) -> str:
    """去掉空白和常见标点后的文本，用于片段比对。"""
    return re.sub(r"[\s，。！？；：、,.!?;:'\"“”‘’（）()\[\]【】]", "", text or "")


def _content_matches(quoted: str, source: str) -> bool:
    """引用片段是否与库中内容匹配：先精确包含，再退化为相似度。"""
    if not quoted:
        return False
    q = _normalize(quoted)
    s = _normalize(source)
    if not q or not s:
        return False
    if q in s or s in q:
        return True
    return difflib.SequenceMatcher(None, q, s).ratio() >= 0.75


def _format_items(items: Sequence[str]) -> str:
    items = [str(x) for x in items]
    shown = items[:MAX_FEEDBACK_ITEMS]
    lines = "\n".join(f"- {x}" for x in shown)
    if len(items) > MAX_FEEDBACK_ITEMS:
        lines += f"\n- ……（共 {len(items)} 条，仅列出前 {MAX_FEEDBACK_ITEMS} 条）"
    return lines


# ---------------------------------------------------------------------------
# 缺字段判定
# ---------------------------------------------------------------------------
def missing_required_fields(parsed: StructuredAnswer | None) -> list[str]:
    """返回缺失的必填内容（人话描述）；空列表表示结构完整。

    规则：
    - natural_language_reply 必填；
    - citations 仅允许在 status.no_support == True 时为空；
    - status.conflict == True 时必须给出引用（证明冲突的是什么）。
    """
    if parsed is None:
        return ["结构化解析结果（Model B 未能成功解析）"]

    missing: list[str] = []
    reply = (parsed.natural_language_reply or "").strip()
    if not reply:
        missing.append("natural_language_reply（给用户的自然语言回复）")

    status = parsed.status
    if status.conflict and not parsed.citations:
        missing.append("citations（文档冲突时，需要引用冲突双方的原文）")
    elif not parsed.citations and not status.no_support:
        missing.append("citations（引用文档 ID 与原文片段）")

    return missing


# ---------------------------------------------------------------------------
# 引用校验
# ---------------------------------------------------------------------------
def validate_citations(
    citations: Sequence[Any],
    library: tuple[list[dict], list[dict]] | None = None,
) -> list[str]:
    """逐条检查引用是否与库对得上，返回人话错误列表。

    校验口径（确定性的事交给代码）：
    - document_id 必须存在于库中；
    - quoted_content 必须能在该文档的任意片段（单句或小段落）里找到。

    ``library`` 参数便于测试注入；默认懒加载 data/ 下的 solid 数据。
    库不存在时返回空列表（没有可校验的依据，不阻塞流程）。
    """
    if library is None:
        library = load_solid_library()
    if library is None:
        return []

    sentences, chunks = library
    contents_by_doc: dict[int, list[str]] = {}
    for record in list(sentences) + list(chunks):
        doc_id = record.get("id")
        if doc_id is None:
            continue
        contents_by_doc.setdefault(doc_id, []).append(record.get("content", ""))

    errors: list[str] = []
    for citation in citations:
        document_id = getattr(citation, "document_id", None)
        quoted = getattr(citation, "quoted_content", "") or ""
        where = f"【document_id={document_id}】"

        if document_id not in contents_by_doc:
            errors.append(f"引用{where}不存在，库中没有这个文档 ID")
            continue

        if not any(
            _content_matches(quoted, content) for content in contents_by_doc[document_id]
        ):
            errors.append(
                f"引用{where}的原文片段在该文档中找不到，请核对引用的内容是否逐字来自该文档"
            )

    return errors


# ---------------------------------------------------------------------------
# 反馈消息构造
# ---------------------------------------------------------------------------
def make_feedback(kind: str, detail: Any = "") -> HumanMessage:
    """构造一条带 guard 标记的 HumanMessage。

    kind ∈ {missing_fields, bad_refs, unsupported, finalize}
    """
    if kind == "missing_fields":
        text = FEEDBACK_MISSING_FIELDS.format(detail=_format_items(detail))
    elif kind == "bad_refs":
        text = FEEDBACK_BAD_REFS.format(detail=_format_items(detail))
    elif kind == "unsupported":
        text = FEEDBACK_UNSUPPORTED.format(detail=str(detail).strip() or "（未给出具体理由）")
    elif kind == "finalize":
        text = FEEDBACK_FINALIZE
    else:
        raise ValueError(f"未知的反馈类型：{kind}")

    return HumanMessage(
        content=text,
        id=f"guard-{uuid.uuid4()}",
        additional_kwargs={"guard_feedback": True, "guard_kind": kind},
    )
