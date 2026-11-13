"""Guard 层节点工厂。

用法::

    nodes = make_guard_nodes(
        main_model=main_model,
        main_model_with_tools=main_model_with_tools,
        main_system_prompt=system_prompt,
        tools_by_name=tools_by_name,
    )

返回的节点里，``call_main_llm`` / ``tool_node`` 用普通返回值，
其余节点返回 ``Command(goto=...)`` 自行决定下一步。
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from guard_prompts import (
    MODEL_B_SYSTEM_PROMPT,
    MODEL_C_SYSTEM_PROMPT,
    STEP_LIMIT_MESSAGE,
    TOOL_RESULT_FIELD_DOC,
)
from guard_schemas import StructuredAnswer, SupportVerdict
from guard_validators import (
    content_to_text,
    make_feedback,
    missing_required_fields,
    validate_citations,
)

# 每一步最多往返次数（super-step 近似值）
MAX_STEPS = 100

# Model C 解析失败时的硬上限（"直到成功"需要兜底）
MAX_SUPPORT_PARSE_ATTEMPTS = 5

# Model B 缺字段时的重试次数（首次 + 重试 1 次）
MAX_PARSE_ATTEMPTS = 2

# 单条工具证据截断长度
_TOOL_EVIDENCE_LIMIT = 4000


# ---------------------------------------------------------------------------
# 默认依赖（延迟导入，避免 import 时就触发模型下载 / 数据读取）
# ---------------------------------------------------------------------------
def _default_hash_check() -> bool:
    from assign_ids_and_versiondate import do_we_have_file_change

    return do_we_have_file_change()


def _default_index_ready() -> None:
    from get_index_and_save import get_index_ready

    get_index_ready()


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def over_limit(state: Any) -> bool:
    return (state.get("step_count") or 0) >= MAX_STEPS


def _truncate(text: str, limit: int = _TOOL_EVIDENCE_LIMIT) -> str:
    """保留头尾的截断（当前仅用于调试，证据包已改用全文）。"""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n……（中间省略 {len(text) - limit} 字）……\n{text[-tail:]}"


def _tool_message(content: Any, tool_call: dict, name: str | None) -> ToolMessage:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    return ToolMessage(
        content=content,
        tool_call_id=tool_call.get("id", ""),
        name=name,
        id=f"tool-{uuid.uuid4()}",
    )


def build_evidence_pack(state: Any) -> str:
    """把 Model C 需要判断的材料打包成一段文本。"""
    lines: list[str] = ["[本轮用户问题]", state.get("user_question") or "（未识别）", ""]
    lines.append("[文档检索记录]")

    tool_messages = [m for m in state.get("messages", []) if isinstance(m, ToolMessage)]
    if not tool_messages:
        lines.append("（本轮没有工具检索记录）")
    for i, msg in enumerate(tool_messages, 1):
        lines.append(f"{i}. 工具：{getattr(msg, 'name', None) or '未知'}")
        # 证据包保留工具返回的完整内容（不截断），保证 Model C 看到的是全文
        lines.append("   结果：" + content_to_text(msg.content))

    lines += ["", "[待判定的最终回复]"]
    last_ai: AIMessage | None = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            last_ai = msg
            break
    lines.append(content_to_text(last_ai.content) if last_ai is not None else "（无）")
    lines += ["", TOOL_RESULT_FIELD_DOC, "", "请判断：上述文档内容是否完全支持该回复中的结论？"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 节点工厂
# ---------------------------------------------------------------------------
def make_guard_nodes(
    *,
    main_model: Any,
    main_model_with_tools: Any,
    main_system_prompt: str,
    tools_by_name: dict[str, Any] | None = None,
    hash_check_fn: Callable[[], bool] | None = None,
    index_ready_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """创建主图所需的全部节点（含 Model A 与 Guard 链）。"""
    tools_by_name = tools_by_name or {}
    hash_check_fn = hash_check_fn or _default_hash_check
    index_ready_fn = index_ready_fn or _default_index_ready

    # Model B / Model C：与 Model A 共用底层模型，绑定不同 schema
    model_b = main_model.with_structured_output(StructuredAnswer)
    model_c = main_model.with_structured_output(SupportVerdict)

    # -- START 初始化 -------------------------------------------------------
    async def init_turn(state: Any) -> Command:
        update: dict[str, Any] = {
            "step_count": 1,
            "phase": "draft",
            "debate_flag": False,
            "feedback_injected": False,
            "parsed_answer": None,
            "parse_attempts": 0,
            "support_verdict": None,
            "support_attempts": 0,
            "evidence_pack": None,
            "hash_changed": False,
            "history_cleared": False,
            "index_rebuild_error": None,
        }

        # 1) hash 检查
        changed = False
        try:
            changed = bool(hash_check_fn())
        except Exception as exc:  # noqa: BLE001 - 不因 hash 检查失败而中断问答
            update["index_rebuild_error"] = f"hash 检查失败：{type(exc).__name__}: {exc}"

        # 2) 定位本轮真实提问：从后往前，跳过 guard 反馈与上一轮提问
        prev_id = state.get("user_question_msg_id")
        current: HumanMessage | None = None
        for message in reversed(state.get("messages", [])):
            if not isinstance(message, HumanMessage):
                continue
            if (message.additional_kwargs or {}).get("guard_feedback"):
                continue
            if prev_id and message.id == prev_id:
                break
            current = message
            break

        if current is None:
            update["messages"] = [
                AIMessage(content="没有检测到新的用户提问，本轮先不处理。")
            ]
            return Command(goto=END, update=update)

        update["user_question"] = content_to_text(current.content)
        update["user_question_msg_id"] = current.id
        update["hash_changed"] = changed

        # 3) hash 变了：清空历史，只保留本轮提问，并重建索引
        if changed:
            update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), current]
            update["history_cleared"] = True
            try:
                index_ready_fn()
            except Exception as exc:  # noqa: BLE001 - 索引重建失败不阻塞问答
                update["index_rebuild_error"] = (
                    f"索引重建失败：{type(exc).__name__}: {exc}"
                )

        return Command(goto="call_main_llm", update=update)

    # -- Model A ------------------------------------------------------------
    async def call_main_llm(state: Any) -> dict[str, Any]:
        feed = [SystemMessage(content=main_system_prompt)] + list(state.get("messages", []))
        response = await main_model_with_tools.ainvoke(feed)
        return {"messages": [response], "step_count": 1}

    # -- 工具节点（执行完回到 call_main_llm）---------------------------------
    async def tool_node(state: Any) -> dict[str, Any]:
        result: list[ToolMessage] = []
        last = state["messages"][-1]
        for tool_call in getattr(last, "tool_calls", None) or []:
            name = tool_call.get("name")
            tool = tools_by_name.get(name)
            if tool is None:
                result.append(
                    _tool_message(
                        "tool call invalid：该工具当前不可用，请换一个工具或直接回答。",
                        tool_call,
                        name,
                    )
                )
                continue
            try:
                observation = await tool.ainvoke(tool_call.get("args", {}))
                result.append(_tool_message(observation, tool_call, name))
            except Exception as exc:  # noqa: BLE001 - 工具异常转成 ToolMessage
                result.append(_tool_message(str(exc), tool_call, name))
        return {"messages": result, "step_count": 1}

    # -- Guard 1：解析 + 引用校验 --------------------------------------------
    async def guard_parse(state: Any) -> Command:
        if over_limit(state):
            return Command(goto="step_limit", update={"step_count": 1})

        draft = state["messages"][-1]
        if not isinstance(draft, AIMessage) or getattr(draft, "tool_calls", None):
            # 防御：不是最终回答，不应到这里
            return Command(goto=END, update={"step_count": 1})

        text = content_to_text(draft.content)

        parsed: StructuredAnswer | None = None
        attempts = 0
        for _ in range(MAX_PARSE_ATTEMPTS):
            attempts += 1
            try:
                parsed = await model_b.ainvoke(
                    [SystemMessage(content=MODEL_B_SYSTEM_PROMPT), HumanMessage(content=text)]
                )
            except Exception:  # noqa: BLE001 - 解析失败按缺字段处理
                parsed = None
            if parsed is not None and not missing_required_fields(parsed):
                break

        missing = missing_required_fields(parsed)
        if missing:
            return Command(
                goto="feedback",
                update={
                    "messages": [make_feedback("missing_fields", missing)],
                    "parse_attempts": attempts,
                    "step_count": 1,
                },
            )

        errors = validate_citations(parsed.citations)  # type: ignore[union-attr]
        if errors:
            return Command(
                goto="feedback",
                update={
                    "messages": [make_feedback("bad_refs", errors)],
                    "parsed_answer": parsed.model_dump(),  # type: ignore[union-attr]
                    "parse_attempts": attempts,
                    "step_count": 1,
                },
            )

        return Command(
            goto="guard_support",
            update={
                "parsed_answer": parsed.model_dump(),  # type: ignore[union-attr]
                "parse_attempts": attempts,
                "step_count": 1,
            },
        )

    # -- Guard 2：支持性判定（Model C）--------------------------------------
    async def guard_support(state: Any) -> Command:
        if over_limit(state):
            return Command(goto="step_limit", update={"step_count": 1})

        pack = build_evidence_pack(state)
        verdict: SupportVerdict | None = None
        attempts = 0
        for _ in range(MAX_SUPPORT_PARSE_ATTEMPTS):
            attempts += 1
            try:
                verdict = await model_c.ainvoke(
                    [SystemMessage(content=MODEL_C_SYSTEM_PROMPT), HumanMessage(content=pack)]
                )
            except Exception:  # noqa: BLE001 - 解析失败就重试
                verdict = None
            if verdict is not None:
                break

        if verdict is None:
            return Command(
                goto="feedback",
                update={
                    "messages": [
                        make_feedback(
                            "unsupported",
                            "系统暂时无法确认依据（判定结果解析失败），先按证据不足处理。",
                        )
                    ],
                    "support_attempts": attempts,
                    "debate_flag": True,
                    "evidence_pack": pack,
                    "step_count": 1,
                },
            )

        if verdict.support == "unsupported":
            return Command(
                goto="feedback",
                update={
                    "messages": [make_feedback("unsupported", verdict.reason)],
                    "support_verdict": verdict.model_dump(),
                    "support_attempts": attempts,
                    "debate_flag": True,
                    "evidence_pack": pack,
                    "step_count": 1,
                },
            )

        # 两 Guard 都一次过：Model A 的最后一条消息就是最终答复，直接结束；
        # 只要本轮注入过任何反馈，才需要让 Model A 整理成面向用户的最终回复。
        # finalize 之后不再做任何校验。
        base_update: dict[str, Any] = {
            "support_verdict": verdict.model_dump(),
            "support_attempts": attempts,
            "evidence_pack": pack,
            "step_count": 1,
        }
        if state.get("feedback_injected"):
            return Command(goto="finalize", update=base_update)
        return Command(goto=END, update=base_update)

    # -- 辩论后的最终整理 ----------------------------------------------------
    async def finalize(state: Any) -> Command:
        if over_limit(state):
            return Command(goto="step_limit", update={"step_count": 1})
        return Command(
            goto="call_main_llm",
            update={
                "messages": [make_feedback("finalize")],
                "phase": "final",
                "step_count": 1,
            },
        )

    # -- 反馈注入（唯一回 Model A 的入口）-----------------------------------
    async def feedback(state: Any) -> Command:
        if over_limit(state):
            return Command(goto="step_limit", update={"step_count": 1})
        return Command(
            goto="call_main_llm",
            update={"phase": "draft", "feedback_injected": True, "step_count": 1},
        )

    # -- 步数上限 ------------------------------------------------------------
    async def step_limit(state: Any) -> Command:
        return Command(
            goto=END,
            update={
                "messages": [
                    AIMessage(content=STEP_LIMIT_MESSAGE.format(max_steps=MAX_STEPS))
                ],
                "step_count": 1,
            },
        )

    return {
        "init_turn": init_turn,
        "call_main_llm": call_main_llm,
        "tool_node": tool_node,
        "guard_parse": guard_parse,
        "guard_support": guard_support,
        "finalize": finalize,
        "feedback": feedback,
        "step_limit": step_limit,
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
def route_after_model_a(state: Any) -> str:
    """Model A 执行后的路由：工具循环 / 最终整理 / Guard 1 / 步数上限。"""
    if over_limit(state):
        return "step_limit"
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_node"
    if state.get("phase") == "final":
        return END            # finalize 后的最终回复：不再做任何校验
    return "guard_parse"
