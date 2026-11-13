"""Trace 中 Guard 结构化结果的记录测试。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from knowledge_agent.tracing import TraceRecorder


def _recorder() -> TraceRecorder:
    recorder = TraceRecorder(
        case_id="case",
        title="title",
        mode="online",
        model={"name": "deepseek-chat"},
    )
    recorder.start_turn(1, "问题")
    return recorder


def _parsed() -> dict:
    return {
        "natural_language_reply": "答案",
        "citations": [{"document_id": 1, "quoted_content": "其他城市每晚不超过 350 元。"}],
        "gaps": ["缺少职级档位"],
        "status": {"done": True, "conflict": False, "no_support": False},
    }


def test_guard_pass_records_structured_answer_and_status():
    recorder = _recorder()
    recorder.add_node_event(
        "guard_parse", {"parsed_answer": _parsed(), "parse_attempts": 1}
    )
    recorder.add_node_event(
        "guard_support",
        {
            "support_verdict": {
                "support": "support",
                "error_type": "none",
                "reason": "文档支持",
            },
            "support_attempts": 1,
            "evidence_pack": "证据",
        },
    )
    recorder.add_event("call_main_llm", AIMessage(content="答案"))
    turn = recorder.end_turn(
        {"step_count": 6, "feedback_injected": False, "debate_flag": False}
    )

    assert turn["task_status"]["guard1"] == "pass"
    assert turn["task_status"]["guard2"] == "support"
    assert turn["task_status"]["label"] == "已完成"
    assert turn["citations"][0]["document_id"] == 1
    assert turn["gaps"] == ["缺少职级档位"]

    payload = recorder.status_payload()
    assert "已完成" in payload["content"]
    assert "D001" in payload["content"]
    assert "资料缺口" in payload["content"]
    assert "Guard 1：通过" in payload["content"]

    markdown = recorder.to_markdown()
    assert "结构化任务状态" in markdown
    assert "D001" in markdown


def test_guard_feedback_marks_guard_failed():
    recorder = _recorder()
    recorder.add_node_event("guard_parse", {})
    recorder.add_event(
        "guard_parse",
        HumanMessage(
            content="引用对不上",
            additional_kwargs={"guard_feedback": True, "guard_kind": "bad_refs"},
        ),
    )
    recorder.add_event("call_main_llm", AIMessage(content="修正后的答案"))
    turn = recorder.end_turn(
        {"step_count": 5, "feedback_injected": True, "debate_flag": False}
    )

    assert turn["task_status"]["guard1"] == "feedback"
    assert turn["task_status"]["guard2"] == "not_run"
    assert turn["feedback_injected"] is True
    payload = recorder.status_payload()
    assert "Guard 1：未通过" in payload["content"]


def test_unsupported_verdict_is_recorded():
    recorder = _recorder()
    recorder.add_node_event(
        "guard_parse", {"parsed_answer": _parsed(), "parse_attempts": 1}
    )
    recorder.add_node_event(
        "guard_support",
        {
            "support_verdict": {
                "support": "unsupported",
                "error_type": "overclaim",
                "reason": "缺少冲突说明",
            },
            "support_attempts": 1,
        },
    )
    recorder.add_event("call_main_llm", AIMessage(content="草稿"))
    turn = recorder.end_turn(
        {"step_count": 7, "feedback_injected": True, "debate_flag": True}
    )
    assert turn["task_status"]["guard2"] == "unsupported"
    assert turn["support_verdict"]["reason"] == "缺少冲突说明"
