"""RAG 工具：检索、上下文、全文、版本日期。

设计要点：
- 资源懒加载，import 本模块不会加载模型 / 读大文件；
- 同一会话（LangGraph ``thread_id``）内的翻页状态相互隔离；
- 所有可预期异常都转成给模型看的字符串，而不是让图崩掉。
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from .resources import (
    QUERY_PREFIX,
    RagDataError,
    get_embedding_model,
    load_resources,
    resolve_document_path,
)

# 每个 thread 的"同一 query 已翻到第几条"，避免不同会话互相污染。
_QUERY_STATE: dict[str, dict[str, int]] = {}
_MAX_SESSIONS = 128

TOOL_NAMES = [
    "query_the_docs",
    "retrieve_piece_context",
    "read_full_document",
    "get_version_date",
]


# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------
def _session_key(config: RunnableConfig | None) -> str:
    if config:
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if thread_id:
            return str(thread_id)
    return "default"


def reset_query_state(thread_id: str | None = None) -> None:
    """清空某个会话（或全部会话）的翻页状态。索引重建后调用。"""
    if thread_id is None:
        _QUERY_STATE.clear()
    else:
        _QUERY_STATE.pop(str(thread_id), None)


def _query_history(config: RunnableConfig | None) -> dict[str, int]:
    key = _session_key(config)
    if key not in _QUERY_STATE and len(_QUERY_STATE) >= _MAX_SESSIONS:
        # 简单的 FIFO 淘汰，防止长进程里无限增长。
        _QUERY_STATE.pop(next(iter(_QUERY_STATE)))
    return _QUERY_STATE.setdefault(key, {})


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
def _merge_hits(resources: Any, sentence_ids: list[int], chunk_ids: list[int]) -> list[dict]:
    """合并单句 / 小段落结果；单句若已被某个命中段落包含则去重。"""
    hits: list[dict] = []
    seen_contents: set[str] = set()

    for piece_id in chunk_ids:
        if not 0 <= piece_id < len(resources.chunks):
            continue
        record = resources.chunks[piece_id]
        content = record["content"]
        if content in seen_contents:
            continue
        seen_contents.add(content)
        hits.append(
            {
                "document_id": record["id"],
                "piece_content": content,
                "piece_id": piece_id,
                "piece_type": "chunk",
            }
        )

    for piece_id in sentence_ids:
        if not 0 <= piece_id < len(resources.sentences):
            continue
        record = resources.sentences[piece_id]
        content = record["content"]
        if content in seen_contents:
            continue
        if any(content in hit["piece_content"] for hit in hits):
            continue
        seen_contents.add(content)
        hits.append(
            {
                "document_id": record["id"],
                "piece_content": content,
                "piece_id": piece_id,
                "piece_type": "sentence",
            }
        )

    return hits


async def _raw_query(
    resources: Any, query_str: str, already_num: int, new_num: int
) -> list[dict] | str:
    if new_num <= 0:
        return "number_of_retrieval 必须大于 0。"

    total_sentences = resources.index_sentences.ntotal
    total_chunks = resources.index_chunks.ntotal
    depth_s = min(already_num + new_num, total_sentences)
    depth_c = min(already_num + new_num, total_chunks)
    if depth_s <= already_num and depth_c <= already_num:
        return "没有（更多的）匹配项。已经到达检索结果末尾。"

    model = get_embedding_model()
    query_vector = await asyncio.to_thread(
        model.encode,
        [QUERY_PREFIX + query_str],
        normalize_embeddings=True,
    )

    s_ids: list[int] = []
    c_ids: list[int] = []
    if depth_s > 0:
        _, s_ids_raw = resources.index_sentences.search(query_vector, depth_s)
        s_ids = [int(i) for i in s_ids_raw[0][already_num:depth_s] if i >= 0]
    if depth_c > 0:
        _, c_ids_raw = resources.index_chunks.search(query_vector, depth_c)
        c_ids = [int(i) for i in c_ids_raw[0][already_num:depth_c] if i >= 0]

    hits = _merge_hits(resources, s_ids, c_ids)
    if not hits:
        return "没有（更多的）匹配项。"
    return hits


@tool
async def query_the_docs(
    query_str: Annotated[str, "要在文档库里面查询的内容，尽量使用用户问题中的关键词"],
    number_of_retrieval: Annotated[
        int,
        "打算返回的条数，实际返回条数在这个数的一倍和两倍之间"
        "（匹配文档不足时可能少于这个数）",
    ],
    first_try_on_same_query: Annotated[
        bool,
        "是否是某个查询内容的第一次查询。如果传入 False，会自动扩大召回条数并去重："
        "例如第一次查询“x”返回 4 条，第二次传入 False 且 number_of_retrieval=6，"
        "则实际检索 10 条并去掉前 4 条后返回。传入 False 要求查询内容完全一样。"
        "默认 True；对已经查询过的内容再次传 True 会重置累计查询次数。",
    ] = True,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> str | list[dict]:
    """使用 RAG 方式返回文档库中最匹配的若干条内容。

    会从单句和小段落两个层级匹配，并自动去掉被命中段落包含的单句，
    所以最终条数在选择条数的一倍到两倍之间（文档充足时）。
    返回 list[dict]，字段含义：
      document_id：这条内容出自哪个文档（文档 ID）；
      piece_content：命中的原文片段；
      piece_id：该片段在库中的序号（只用于 retrieve_piece_context）；
      piece_type：sentence（单句）或 chunk（小段落）。
    """
    try:
        resources = load_resources()
    except RagDataError as exc:
        return f"工具调用失败：{exc}"

    history = _query_history(config)
    if first_try_on_same_query:
        already = 0
    else:
        if query_str not in history:
            return "这条检索内容是第一次检索，不应该在 first_try_on_same_query 传入 False。"
        already = history[query_str]

    try:
        result = await _raw_query(resources, query_str, already, int(number_of_retrieval))
    except Exception as exc:  # noqa: BLE001 - 工具失败要交给模型处理
        return f"工具调用失败，异常：{type(exc).__name__}: {exc}"

    if isinstance(result, list):
        history[query_str] = already + int(number_of_retrieval)
    return result


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------
def _context_window(solid_list: list[dict], piece_id: int, count: int) -> list[str] | str:
    """取 ``piece_id`` 前后各若干条同文档片段，返回带边界提示的列表。"""
    if not 0 <= piece_id < len(solid_list):
        return "piece_id 超出范围，请复核。"
    count = max(0, int(count))
    doc_id = solid_list[piece_id]["id"]

    start = piece_id
    while start > 0 and solid_list[start - 1]["id"] == doc_id:
        start -= 1
    end = piece_id
    while end + 1 < len(solid_list) and solid_list[end + 1]["id"] == doc_id:
        end += 1

    head_pool = list(range(start, piece_id))
    tail_pool = list(range(piece_id + 1, end + 1))

    half = count // 2
    selected_head = head_pool[-half:] if half else []
    selected_tail = tail_pool[: max(0, count - len(selected_head))]

    # 一侧不够时，从另一侧补齐。
    if len(selected_head) + len(selected_tail) < count:
        extra = count - len(selected_head) - len(selected_tail)
        remaining_head = head_pool[: len(head_pool) - len(selected_head)]
        take_head = min(extra, len(remaining_head))
        if take_head:
            selected_head = remaining_head[-take_head:] + selected_head
            extra -= take_head
        if extra:
            remaining_tail = tail_pool[len(selected_tail):]
            selected_tail = selected_tail + remaining_tail[:extra]

    head_marker = "（已到达原文档开头）" if start == 0 and (
        not selected_head or selected_head[0] == 0
    ) else "（原文档前面还有内容）"
    tail_marker = "（已到达原文档末尾）" if end == len(solid_list) - 1 and (
        not selected_tail or selected_tail[-1] == end
    ) else "（原文档后面还有内容）"

    window = [solid_list[i]["content"] for i in selected_head]
    window.append(solid_list[piece_id]["content"])
    window += [solid_list[i]["content"] for i in selected_tail]
    return [head_marker] + window + [tail_marker]


@tool
async def retrieve_piece_context(
    piece_id: Annotated[int, "要返回上下文的检索结果 ID"],
    piece_type: Annotated[
        Literal["sentence", "chunk"],
        "要返回上下文的检索结果类型，必须跟 piece_id 对应，只能是 sentence 或 chunk",
    ],
    number_of_context_to_retrieve: Annotated[int, "返回上下文的数量（长度）"],
) -> list[str] | str:
    """查看某个检索结果前后的上下文。

    如果 query_the_docs 返回的内容是零散句子、或需要确认前后文，就调用这个工具。
    传入的 piece_id 和 piece_type 要对应；上下文原则上前后各一半，
    文档开头/结尾时自动调整，并在返回内容里标明是否已到达文档边界。
    """
    try:
        resources = load_resources()
    except RagDataError as exc:
        return f"工具调用失败：{exc}"

    if piece_type == "sentence":
        solid_list = resources.sentences
    elif piece_type == "chunk":
        solid_list = resources.chunks
    else:
        return "piece_type 错误，只能是 sentence 和 chunk 中的一种"
    return _context_window(solid_list, piece_id, number_of_context_to_retrieve)


# ---------------------------------------------------------------------------
# 文档元数据
# ---------------------------------------------------------------------------
def _document_label(meta: dict) -> str:
    return meta.get("title") or meta.get("name") or "未命名文档"


def _lookup_document(document_id: int) -> tuple[dict | None, Any]:
    try:
        resources = load_resources()
    except RagDataError:
        return None, None
    meta = resources.document_meta.get(document_id)
    if meta is None:
        return None, None
    return meta, resolve_document_path(meta)


@tool
async def read_full_document(
    document_id: Annotated[int, "要阅读全文的文档 ID"],
) -> str:
    """阅读文档全文（可能很长）。用于确认完整条款、适用范围或冲突双方的具体规定。"""
    meta, path = _lookup_document(document_id)
    if meta is None or path is None:
        return f"文档 ID {document_id} 不存在，请核对 document_id。"
    try:
        body = path.read_text(encoding="utf-8")
        current_mtime = path.stat().st_mtime
    except OSError as exc:
        return f"读取文档失败：{exc}"

    prefix = ""
    recorded_mtime = meta.get("mtime")
    if recorded_mtime is not None and abs(current_mtime - recorded_mtime) > 1e-6:
        prefix = "（文件在索引生成后发生过变动，内容可能跟检索结果有出入）\n"

    header = (
        f"【文档 {document_id}：{_document_label(meta)}；"
        f"版本日期：{meta.get('version_date') or '未声明'}】\n"
    )
    return prefix + header + body


@tool
async def get_version_date(
    document_id: Annotated[int, "要查阅版本日期的文档 ID"],
) -> str:
    """查阅文档的版本日期。两份文档内容冲突时，用它核对哪份更新、是否存在优先级说明。"""
    meta, _ = _lookup_document(document_id)
    if meta is None:
        return f"文档 ID {document_id} 不存在，请核对 document_id。"
    version = meta.get("version_date")
    label = _document_label(meta)
    if not version:
        return f"文档 ID {document_id}（{label}）没有声明版本日期。"
    return f"文档 ID {document_id}（{label}）的版本日期是 {version}。"


async def get_tools() -> list[Any]:
    """返回绑定给 Model A 的工具列表。"""
    return [query_the_docs, retrieve_piece_context, read_full_document, get_version_date]
