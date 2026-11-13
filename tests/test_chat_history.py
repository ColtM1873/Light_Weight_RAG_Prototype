"""Chat-History 持久化层测试。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from knowledge_agent.chat_history import ConversationStore, safe_thread_name
from knowledge_agent.tracing import TraceRecorder


def _recorder() -> TraceRecorder:
    recorder = TraceRecorder(
        case_id="t1",
        title="测试会话",
        mode="online-web",
        model={"provider": "deepseek", "name": "deepseek-chat"},
    )
    recorder.start_turn(1, "问题")
    recorder.add_event("call_main_llm", AIMessage(content="答案"))
    recorder.end_turn({"step_count": 2, "feedback_injected": False, "debate_flag": False})
    return recorder


def test_append_turn_creates_json_and_markdown(tmp_path):
    store = ConversationStore(tmp_path)
    path = store.append_turn("thread-1", _recorder(), source="web")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["thread_id"] == "thread-1"
    assert len(data["turns"]) == 1
    assert data["turns"][0]["question"] == "问题"
    assert data["turns"][0]["final_answer"] == "答案"
    assert path.with_suffix(".md").exists()
    assert "浏览器会话记录" in path.with_suffix(".md").read_text(encoding="utf-8")


def test_append_twice_keeps_history(tmp_path):
    store = ConversationStore(tmp_path)
    store.append_turn("thread-1", _recorder())
    store.append_turn("thread-1", _recorder())
    data = store.load_thread("thread-1")
    assert data is not None
    assert [turn["turn"] for turn in data["turns"]] == [1, 2]
    index_lines = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index_lines) == 2


def test_error_turn_is_saved(tmp_path):
    store = ConversationStore(tmp_path)
    recorder = _recorder()
    path = store.append_turn("thread-err", recorder, error="RuntimeError: boom")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["turns"][0]["error"] == "RuntimeError: boom"


def test_safe_thread_name_blocks_path_traversal():
    assert "/" not in safe_thread_name("../evil/thread")
    assert "\\" not in safe_thread_name(r"..\evil\thread")
    assert safe_thread_name("abc-123") == "abc-123"


def test_tool_event_records_args(tmp_path):
    store = ConversationStore(tmp_path)
    recorder = TraceRecorder(case_id="t", title="t", mode="m", model={})
    recorder.start_turn(1, "问题")
    recorder.add_event("call_main_llm", AIMessage(
        content="",
        tool_calls=[{"name": "query_the_docs", "args": {"query_str": "x"}, "id": "c1"}],
    ))
    recorder.add_event("tool_node", ToolMessage(
        content="结果", tool_call_id="c1", name="query_the_docs"
    ))
    recorder.add_event("call_main_llm", AIMessage(content="答"))
    recorder.end_turn({"step_count": 4})
    store.append_turn("thread-tool", recorder)
    data = store.load_thread("thread-tool")
    tool_event = [e for e in data["turns"][0]["events"] if e["kind"] == "tool"][0]
    assert tool_event["tool_args"] == {"query_str": "x"}
