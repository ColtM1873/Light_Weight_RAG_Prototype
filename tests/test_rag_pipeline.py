"""RAG 管线 / 上下文窗口 / 故障注入的单元测试（不加载 embedding 模型）。"""

from __future__ import annotations

from knowledge_agent.nodes import ToolFaultPolicy
from knowledge_agent.paths import DATA_DIR, PROJECT_ROOT
from knowledge_agent.rag.chunking import chunk_up, get_sentences
from knowledge_agent.rag.tools import _context_window, query_the_docs


def test_data_dir_is_inside_project_root():
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert DATA_DIR.is_dir()


def test_query_tool_schema_hides_injected_config():
    # config 是注入参数，不应暴露给模型。
    assert "config" not in query_the_docs.args
    assert {"query_str", "number_of_retrieval", "first_try_on_same_query"} <= set(
        query_the_docs.args
    )


def test_get_sentences_splits_chinese_punctuation():
    text = "第一句。第二句，第三句！"
    assert get_sentences(text) == ["第一句。", "第二句，", "第三句！"]


def test_get_sentences_caps_long_sentence():
    long_sentence = "甲" * 1200 + "。"
    pieces = get_sentences(long_sentence)
    assert len(pieces) >= 3
    assert all(len(piece) <= 512 for piece in pieces)


def test_chunk_up_keeps_chunks_under_cap():
    sentences = ["一" * 120 + "。" for _ in range(10)]
    chunks = chunk_up(sentences)
    assert chunks
    assert all(len(chunk) <= 512 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) >= sum(len(s) for s in sentences)


def test_context_window_centers_and_marks_boundaries():
    solid = [{"id": 1, "content": f"a{i}"} for i in range(5)]
    window = _context_window(solid, 0, 2)
    assert window[0] == "（已到达原文档开头）"
    assert "a0" in window
    assert window[-1] == "（原文档后面还有内容）"

    middle = _context_window(solid, 2, 2)
    assert middle[0] == "（原文档前面还有内容）"
    assert middle[-1] == "（原文档后面还有内容）"


def test_context_window_handles_document_end():
    solid = [{"id": 1, "content": f"a{i}"} for i in range(5)]
    window = _context_window(solid, 4, 10)
    assert window[0] == "（已到达原文档开头）"
    assert window[-1] == "（已到达原文档末尾）"


def test_context_window_invalid_piece_id():
    result = _context_window([{"id": 1, "content": "a"}], 3, 2)
    assert isinstance(result, str)
    assert "超出范围" in result


def test_fault_policy_fails_only_configured_times():
    policy = ToolFaultPolicy(tool_name="query_the_docs", times=2)
    assert policy.message_for("other_tool") is None
    assert "模拟故障 1/2" in (policy.message_for("query_the_docs") or "")
    assert "模拟故障 2/2" in (policy.message_for("query_the_docs") or "")
    assert policy.message_for("query_the_docs") is None
