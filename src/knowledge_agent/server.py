"""FastAPI 服务：浏览器里逐条打印 Human / Tool / AI 消息（SSE）。

服务端额外做一层会话持久化：每轮问答结束后自动写入根目录 ``Chat-History/``，
包含完整事件、工具参数/结果、Guard 反馈和最终答复，便于复盘与整理 Trace。

启动方式（推荐）::

    python run.py serve

也可以::

    .venv\\Scripts\\python.exe -m uvicorn knowledge_agent.server:app --host 127.0.0.1 --port 8010
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from .chat_history import CHAT_HISTORY_DIR, ConversationStore
from .config import public_config
from .graph import TheWorld
from .nodes import ToolFaultPolicy
from .paths import STATIC_DIR
from .rag.resources import data_status
from .tracing import TraceRecorder
from .validators import content_to_text

app = FastAPI(title="企业知识问答 Agent · RAG + Guard Demo")

_graph = None
_graph_lock = asyncio.Lock()
_history_store = ConversationStore()

_NODE_ONLY_EVENTS = {"init_turn", "guard_parse", "guard_support", "feedback"}
_GUARD_1_KINDS = {"missing_fields", "bad_refs"}


async def get_graph():
    """整个进程复用一张编译好的图 + 一个 checkpointer（按 thread_id 存历史）。"""
    global _graph
    if _graph is None:
        async with _graph_lock:
            if _graph is None:
                _graph = await TheWorld(
                    InMemorySaver(), fault_policy=ToolFaultPolicy.from_env()
                )
    return _graph


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _serialize(message: BaseMessage) -> dict[str, Any] | None:
    """把进入/离开节点的消息转成前端需要的最小结构。"""
    if isinstance(message, HumanMessage):
        kind, label = "human", "Human"
        guard_kind = (message.additional_kwargs or {}).get("guard_kind")
        if guard_kind:
            label = f"Human · guard:{guard_kind}"
    elif isinstance(message, AIMessage):
        kind, label = "ai", "AI"
    elif isinstance(message, ToolMessage):
        kind, label = "tool", f"Tool · {getattr(message, 'name', None) or 'unknown'}"
    else:
        return None

    content = content_to_text(message.content)
    tool_calls = [
        tool_call.get("name")
        for tool_call in (getattr(message, "tool_calls", None) or [])
        if isinstance(tool_call, dict)
    ]
    if not content and tool_calls:
        content = "→ 调用工具：" + ", ".join(str(name) for name in tool_calls)
    if not content:
        return None

    return {
        "type": kind,
        "label": label,
        "content": content,
        "id": message.id,
    }


def _model_info() -> dict[str, Any]:
    config = public_config()
    return {"provider": config.get("provider"), "name": config.get("model")}


def _guard_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    """把 Guard 节点事件转成前端可读的卡片。"""
    is_parse = event.get("guard") == "parse"
    label = "Guard 1 · 解析与引用校验" if is_parse else "Guard 2 · 证据支持性判定"
    payload: dict[str, Any] = {
        "type": "guard",
        "label": label,
        "ok": bool(event.get("ok")),
        "content": event.get("content") or "",
    }
    if is_parse:
        payload["parsed_answer"] = event.get("parsed_answer")
    else:
        payload["support_verdict"] = event.get("support_verdict")
    return payload


def _guard_feedback_payload(node: str, messages: list[Any]) -> dict[str, Any] | None:
    """Guard 未通过时，除了 Human 反馈消息，再给前端一张简明的 Guard 卡片。"""
    feedback = next(
        (
            message
            for message in messages
            if isinstance(message, HumanMessage)
            and (message.additional_kwargs or {}).get("guard_kind")
        ),
        None,
    )
    if feedback is None:
        return None
    guard_kind = (feedback.additional_kwargs or {}).get("guard_kind")
    label = "Guard 1 · 未通过" if guard_kind in _GUARD_1_KINDS else "Guard 2 · 未通过"
    return {
        "type": "guard",
        "label": label,
        "ok": False,
        "guard_kind": guard_kind,
        "content": f"反馈类型：{guard_kind}\n已把问题回给 Model A 修正。",
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "config": public_config(),
        "data": data_status(),
        "chat_history_dir": str(CHAT_HISTORY_DIR),
    }


@app.get("/api/chat")
async def chat(
    message: str = Query(..., min_length=1),
    thread_id: str = Query(...),
):
    graph = await get_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 120,
    }
    recorder = TraceRecorder(
        case_id=thread_id,
        title="浏览器会话",
        mode="online-web",
        model=_model_info(),
    )
    recorder.start_turn(1, message)

    async def event_stream() -> AsyncIterator[str]:
        error_message: str | None = None
        try:
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode="updates",
            ):
                if not isinstance(chunk, dict):
                    continue
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    messages = update.get("messages") or []
                    for msg in messages:
                        event = recorder.add_event(node, msg)
                        payload = _serialize(msg)
                        if payload is not None:
                            payload["node"] = node
                            yield _sse("message", payload)
                    if not messages and node in _NODE_ONLY_EVENTS:
                        event = recorder.add_node_event(node, update)
                        if event is not None and node in {"guard_parse", "guard_support"}:
                            yield _sse("message", _guard_event_payload(event))
                    if messages and node in {"guard_parse", "guard_support"}:
                        payload = _guard_feedback_payload(node, messages)
                        if payload is not None:
                            yield _sse("message", payload)
        except Exception as exc:  # noqa: BLE001 - 推给前端而不是断连接
            error_message = f"{type(exc).__name__}: {exc}"

        # 无论成功失败，都把本轮写入 Chat-History（失败也保留，便于排查）。
        try:
            snapshot = await graph.aget_state(config)
            state_values = dict(snapshot.values or {})
            recorder.end_turn(state_values)
        except Exception as exc:  # noqa: BLE001
            recorder.end_turn({"step_count": None, "index_rebuild_error": str(exc)})

        # 给前端一张结构化状态卡（任务状态 / 引用 / 缺口 / Guard 结果）。
        yield _sse("message", recorder.status_payload())

        try:
            path = await asyncio.to_thread(
                _history_store.append_turn,
                thread_id,
                recorder,
                source="web",
                error=error_message,
                model=_model_info(),
            )
            yield _sse("saved", {"file": path.name})
        except Exception as exc:  # noqa: BLE001 - 持久化失败不影响主流程
            print(f"[Chat-History] 保存失败：{type(exc).__name__}: {exc}")

        if error_message is not None:
            yield _sse("failed", {"message": error_message})
        else:
            yield _sse("done", {"thread_id": thread_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)


if __name__ == "__main__":
    main()
