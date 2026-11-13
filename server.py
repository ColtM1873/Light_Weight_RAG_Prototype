"""FastAPI 服务：在浏览器里逐步打印 Human / Tool / AI 消息。

启动：
    .venv\\Scripts\\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8010
然后打开 http://127.0.0.1:8010
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from build_the_world import TheWorld
from guard_validators import content_to_text

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Light Weight RAG · Guard Demo")

_graph = None
_graph_lock = asyncio.Lock()


async def get_graph():
    """整个进程复用一张编译好的图 + 一个 checkpointer（按 thread_id 存历史）。"""
    global _graph
    if _graph is None:
        async with _graph_lock:
            if _graph is None:
                _graph = await TheWorld(InMemorySaver())
    return _graph


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _serialize(msg: BaseMessage) -> dict[str, Any] | None:
    """把进入/离开节点的消息转成前端需要的最小结构。"""
    if isinstance(msg, HumanMessage):
        kind, label = "human", "Human"
        guard_kind = (msg.additional_kwargs or {}).get("guard_kind")
        if guard_kind:
            label = f"Human · guard:{guard_kind}"
    elif isinstance(msg, AIMessage):
        kind, label = "ai", "AI"
    elif isinstance(msg, ToolMessage):
        kind, label = "tool", f"Tool · {getattr(msg, 'name', None) or 'unknown'}"
    else:
        return None

    content = content_to_text(msg.content)
    tool_calls = [
        tc.get("name")
        for tc in (getattr(msg, "tool_calls", None) or [])
        if isinstance(tc, dict)
    ]
    if not content and tool_calls:
        content = "→ 调用工具：" + ", ".join(str(name) for name in tool_calls)
    if not content:
        return None

    return {
        "type": kind,
        "label": label,
        "content": content,
        "id": msg.id,
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/chat")
async def chat(
    message: str = Query(..., min_length=1),
    thread_id: str = Query(...),
):
    graph = await get_graph()

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=message)]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 120,
                },
                stream_mode="updates",
            ):
                if not isinstance(chunk, dict):
                    continue
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    for msg in update.get("messages") or []:
                        payload = _serialize(msg)
                        if payload is not None:
                            payload["node"] = node
                            yield _sse("message", payload)
            yield _sse("done", {"thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001 - 推给前端而不是断连接
            yield _sse(
                "failed",
                {"message": f"{type(exc).__name__}: {exc}"},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
