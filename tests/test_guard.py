"""Guard 层单元测试（不依赖真实 LLM / 数据）。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import END, START, StateGraph

from build_the_world import TheWorld  # noqa: F401 - 验证顶层可导入
from guard_nodes import MAX_STEPS, make_guard_nodes, route_after_model_a
from guard_schemas import AnswerStatus, Citation, StructuredAnswer
from guard_validators import (
    content_to_text,
    make_feedback,
    missing_required_fields,
    validate_citations,
)
from variant_message_state import ExtendedMessagesState


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class _StubModel:
    """不真正调用模型：只用于构造节点 / 编译图。"""

    async def ainvoke(self, messages):
        return AIMessage(content="stub")

    def with_structured_output(self, schema):
        return self

    def bind_tools(self, tools, **kwargs):
        return self


def _make_nodes(**overrides):
    kwargs = {
        "main_model": _StubModel(),
        "main_model_with_tools": _StubModel(),
        "main_system_prompt": "test",
        "hash_check_fn": lambda: False,
        "index_ready_fn": lambda: None,
    }
    kwargs.update(overrides)
    return make_guard_nodes(**kwargs)


def _answer(**overrides):
    base = {
        "natural_language_reply": "这是答案",
        "citations": [],
        "gaps": [],
        "status": AnswerStatus(done=True, conflict=False, no_support=False),
    }
    base.update(overrides)
    return StructuredAnswer(**base)


def _citation(**overrides):
    base = {
        "document_id": 1001,
        "quoted_content": "第一段内容",
    }
    base.update(overrides)
    return Citation(**base)


LIBRARY = (
    [{"id": 1001, "content": "第一段内容。", "piece_id": 0}],
    [{"id": 1002, "content": "一个段落内容。", "piece_id": 0}],
)


# ---------------------------------------------------------------------------
# missing_required_fields
# ---------------------------------------------------------------------------
def test_complete_answer_has_no_missing_fields():
    parsed = _answer(citations=[_citation()])
    assert missing_required_fields(parsed) == []


def test_no_support_allows_empty_citations():
    parsed = _answer(
        status=AnswerStatus(done=True, conflict=False, no_support=True)
    )
    assert missing_required_fields(parsed) == []


def test_conflict_requires_citations():
    parsed = _answer(
        status=AnswerStatus(done=True, conflict=True, no_support=False)
    )
    missing = missing_required_fields(parsed)
    assert len(missing) == 1
    assert "citations" in missing[0]


def test_empty_reply_is_missing():
    parsed = _answer(natural_language_reply="   ", citations=[_citation()])
    missing = missing_required_fields(parsed)
    assert any("natural_language_reply" in item for item in missing)


def test_none_answer_is_missing():
    assert missing_required_fields(None)


# ---------------------------------------------------------------------------
# validate_citations
# ---------------------------------------------------------------------------
def test_valid_citation_passes():
    assert validate_citations([_citation()], library=LIBRARY) == []


def test_chunk_content_citation_passes():
    citation = _citation(document_id=1002, quoted_content="一个段落内容")
    assert validate_citations([citation], library=LIBRARY) == []


def test_unknown_document_id_is_caught():
    errors = validate_citations([_citation(document_id=9999)], library=LIBRARY)
    assert len(errors) == 1
    assert "不存在" in errors[0]


def test_quoted_content_mismatch_is_caught():
    errors = validate_citations([_citation(quoted_content="完全不相干的内容")], library=LIBRARY)
    assert len(errors) == 1
    assert "找不到" in errors[0]


def test_empty_library_flags_unknown_document():
    # 显式传空库时，任何 document_id 都被视为不存在
    assert validate_citations([_citation()], library=([], [])) != []


# ---------------------------------------------------------------------------
# make_feedback
# ---------------------------------------------------------------------------
def test_feedback_has_guard_marker():
    msg = make_feedback("missing_fields", ["citations（引用）"])
    assert isinstance(msg, HumanMessage)
    assert msg.additional_kwargs["guard_feedback"] is True
    assert msg.additional_kwargs["guard_kind"] == "missing_fields"
    assert "citations" in msg.content


def test_unknown_feedback_kind_raises():
    try:
        make_feedback("bogus")
    except ValueError:
        return
    raise AssertionError("应当抛出 ValueError")


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
def test_route_tool_calls_goes_to_tool_node():
    state = {
        "messages": [AIMessage(content="", tool_calls=[{"name": "x", "id": "1", "args": {}}])],
        "step_count": 3,
    }
    assert route_after_model_a(state) == "tool_node"


def test_route_plain_answer_goes_to_guard_parse():
    state = {"messages": [AIMessage(content="答案")], "step_count": 3}
    assert route_after_model_a(state) == "guard_parse"


def test_route_over_limit_goes_to_step_limit():
    state = {"messages": [AIMessage(content="答案")], "step_count": MAX_STEPS}
    assert route_after_model_a(state) == "step_limit"


def test_route_final_phase_goes_to_end():
    state = {"messages": [AIMessage(content="最终稿")], "step_count": 3, "phase": "final"}
    assert route_after_model_a(state) == END


# ---------------------------------------------------------------------------
# init_turn
# ---------------------------------------------------------------------------
def test_init_turn_finds_new_question():
    nodes = _make_nodes()
    state = {"messages": [HumanMessage(content="用户问题", id="u1")]}
    cmd = asyncio.run(nodes["init_turn"](state))
    assert cmd.goto == "call_main_llm"
    assert cmd.update["user_question"] == "用户问题"
    assert cmd.update["user_question_msg_id"] == "u1"


def test_init_turn_skips_guard_feedback():
    nodes = _make_nodes()
    state = {
        "messages": [
            HumanMessage(content="上一轮问题", id="u0"),
            AIMessage(content="上一轮回答", id="a0"),
            HumanMessage(
                content="guard 反馈",
                id="g1",
                additional_kwargs={"guard_feedback": True},
            ),
        ],
        "user_question_msg_id": "u0",
    }
    cmd = asyncio.run(nodes["init_turn"](state))
    # 没有新提问：不应把 guard 反馈当成提问
    assert cmd.goto == END
    assert "user_question" not in (cmd.update or {})


def test_init_turn_hash_changed_clears_history():
    nodes = _make_nodes(hash_check_fn=lambda: True)
    state = {
        "messages": [
            HumanMessage(content="新一轮问题", id="u1"),
            AIMessage(content="旧回答", id="a1"),
        ]
    }
    cmd = asyncio.run(nodes["init_turn"](state))
    assert cmd.update["history_cleared"] is True
    messages = cmd.update["messages"]
    assert isinstance(messages[0], RemoveMessage)
    assert messages[0].id == REMOVE_ALL_MESSAGES
    assert len(messages) == 2
    assert messages[1].id == "u1"


def test_init_turn_hash_check_failure_does_not_crash():
    def boom():
        raise RuntimeError("no data")

    nodes = _make_nodes(hash_check_fn=boom)
    state = {"messages": [HumanMessage(content="问题", id="u1")]}
    cmd = asyncio.run(nodes["init_turn"](state))
    assert cmd.goto == "call_main_llm"
    assert "hash 检查失败" in cmd.update["index_rebuild_error"]


# ---------------------------------------------------------------------------
# 图可装配 / 可编译
# ---------------------------------------------------------------------------
def test_graph_compiles():
    nodes = _make_nodes()
    builder = StateGraph(ExtendedMessagesState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "init_turn")
    builder.add_edge("tool_node", "call_main_llm")
    builder.add_conditional_edges(
        "call_main_llm",
        route_after_model_a,
        {
            "tool_node": "tool_node",
            "guard_parse": "guard_parse",
            "step_limit": "step_limit",
            END: END,
        },
    )
    graph = builder.compile()
    assert graph is not None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def test_content_to_text_handles_blocks():
    assert content_to_text("abc") == "abc"
    assert "hi" in content_to_text([{"type": "text", "text": "hi"}])


def test_evidence_pack_mentions_no_tools():
    from guard_nodes import build_evidence_pack

    pack = build_evidence_pack(
        {
            "messages": [AIMessage(content="最终回答")],
            "user_question": "问题",
        }
    )
    assert "本轮没有工具检索记录" in pack
    assert "最终回答" in pack


# ---------------------------------------------------------------------------
# 集成：完整图流转（脚本化模型，不调用真实 LLM）
# ---------------------------------------------------------------------------
from build_the_world import assemble_graph
from guard_schemas import SupportVerdict


class _ScriptedStructured:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    async def ainvoke(self, messages):
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return item


class _ScriptedModel:
    def __init__(self, answers, b_script, c_script):
        self.answers = list(answers)
        self.ai = 0
        self.b_script = b_script
        self.c_script = c_script

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema):
        if schema is StructuredAnswer:
            return _ScriptedStructured(self.b_script)
        return _ScriptedStructured(self.c_script)

    async def ainvoke(self, messages):
        answer = self.answers[min(self.ai, len(self.answers) - 1)]
        self.ai += 1
        return AIMessage(content=answer)


OK_ANSWER = StructuredAnswer(
    natural_language_reply="答案",
    citations=[],
    status=AnswerStatus(done=True, conflict=False, no_support=True),
)
BAD_ANSWER = StructuredAnswer(
    natural_language_reply="答案",
    citations=[],
    status=AnswerStatus(done=True, conflict=False, no_support=False),
)
SUPPORT = SupportVerdict(support="support", error_type="none", reason="文档支持")
UNSUPPORT = SupportVerdict(support="unsupported", error_type="overclaim", reason="证据不足")


def _run_graph(answers, b_script, c_script, monkeypatch):
    import guard_nodes

    monkeypatch.setattr(guard_nodes, "validate_citations", lambda citations: [])
    model = _ScriptedModel(answers, b_script, c_script)
    nodes = make_guard_nodes(
        main_model=model,
        main_model_with_tools=model,
        main_system_prompt="test",
        hash_check_fn=lambda: False,
        index_ready_fn=lambda: None,
    )
    graph = assemble_graph(nodes)
    return asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="问题", id="u1")]}))


def test_happy_path_ends_directly(monkeypatch):
    """两个 Guard 都一次过：不注入 finalize，直接 END。"""
    out = _run_graph(["最终答案"], [OK_ANSWER], [SUPPORT], monkeypatch)
    assert out["debate_flag"] is False
    assert out["feedback_injected"] is False
    assert out["messages"][-1].content == "最终答案"
    assert not any(
        getattr(m, "additional_kwargs", {}).get("guard_kind") == "finalize"
        for m in out["messages"]
    )


def test_missing_fields_triggers_feedback(monkeypatch):
    out = _run_graph(
        ["第一次答案", "补充后的答案", "最终整理稿"],
        [BAD_ANSWER, BAD_ANSWER, OK_ANSWER],
        [SUPPORT],
        monkeypatch,
    )
    assert out["messages"][-1].content == "最终整理稿"
    assert out["feedback_injected"] is True
    assert any(
        getattr(m, "additional_kwargs", {}).get("guard_kind") == "missing_fields"
        for m in out["messages"]
    )


def test_debate_triggers_finalize(monkeypatch):
    out = _run_graph(
        ["答一", "答二", "最终整理稿"],
        [OK_ANSWER, OK_ANSWER, OK_ANSWER],
        [UNSUPPORT, SUPPORT],
        monkeypatch,
    )
    assert out["debate_flag"] is True
    assert out["feedback_injected"] is True
    assert out["messages"][-1].content == "最终整理稿"
    assert any(
        getattr(m, "additional_kwargs", {}).get("guard_kind") == "finalize"
        for m in out["messages"]
    )
