"""统一命令行入口：serve / ask / demo / build-index / check。

示例：
    python run.py check
    python run.py build-index
    python run.py serve
    python run.py ask -q "出差住宿标准是多少？" --trace examples/cases/01.json
    python run.py demo --case conflict
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .config import load_agent_specs, public_config
from .graph import TheWorld, build_graph_with_model
from .nodes import ToolFaultPolicy
from .paths import ENV_FILE, PROJECT_ROOT
from .rag import (
    build_index,
    data_status,
    embedding_dimension,
    ensure_embedding_model,
    get_tools,
)
from .rag.documents import do_we_have_file_change
from .rag.resources import EMBEDDING_MODEL_NAME
from .tracing import TraceRecorder


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + f"\n…（已截断，共 {len(text)} 字）"


def _print_event(event: dict[str, Any]) -> None:
    label = event.get("label") or event.get("node") or ""
    kind = event.get("kind")
    content = event.get("content") or ""
    if kind == "tool":
        content = _clip(content, 1200)
    print(f"\n[{label}]")
    print(content)
    for call in event.get("tool_calls") or []:
        print(f"  ↳ 参数：{json.dumps(call.get('args', {}), ensure_ascii=False)}")


# ---------------------------------------------------------------------------
# 运行图
# ---------------------------------------------------------------------------
async def _run_turns(
    graph: Any,
    questions: Sequence[str],
    thread_id: str,
    recorder: TraceRecorder,
) -> TraceRecorder:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 120,
    }
    for index, question in enumerate(questions, 1):
        print(f"\n{'=' * 72}\nTurn {index} | {question}\n{'=' * 72}")
        recorder.start_turn(index, question)
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="updates",
        ):
            if not isinstance(chunk, dict):
                continue
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                messages = update.get("messages") or []
                for message in messages:
                    event = recorder.add_event(node, message)
                    if event is not None:
                        _print_event(event)
                if not messages and node in {
                    "init_turn",
                    "guard_parse",
                    "guard_support",
                    "feedback",
                }:
                    event = recorder.add_node_event(node, update)
                    if event is not None:
                        print(f"\n[节点] {node}：{event['content']}")

        snapshot = await graph.aget_state(config)
        turn = recorder.end_turn(dict(snapshot.values or {}))
        print(
            f"\n--- Turn {index} 状态：步数={turn['step_count']}，"
            f"工具调用={turn['tool_calls']}，完成={turn['completed']}，"
            f"Guard反馈={turn['feedback_injected']}，辩论={turn['debate_flag']}"
        )
        status = recorder.status_payload()
        print(f"\n[任务状态]\n{status['content']}")
        if turn.get("index_rebuild_error"):
            print(f"索引错误：{turn['index_rebuild_error']}")
    return recorder


def _write_trace(recorder: TraceRecorder, trace: str | None) -> None:
    if not trace:
        return
    target = Path(trace)
    json_path = target if target.suffix.lower() == ".json" else target.with_suffix(".json")
    md_path = json_path.with_suffix(".md")
    recorder.write_json(json_path)
    recorder.write_markdown(md_path)
    print(f"\nTrace 已写入：{json_path}")
    print(f"可读记录：{md_path}")


# ---------------------------------------------------------------------------
# ask（真实模型）
# ---------------------------------------------------------------------------
async def cmd_ask(args: argparse.Namespace) -> int:
    try:
        specs = load_agent_specs()
    except RuntimeError as exc:
        print(f"[配置错误] {exc}")
        return 2

    thread_id = args.thread_id or f"cli-{uuid.uuid4().hex[:8]}"
    fault_policy = None
    if args.fault_tool:
        fault_policy = ToolFaultPolicy(
            tool_name=args.fault_tool, times=max(0, args.fault_times)
        )

    print(f"模型：{specs.get('model_provider')} / {specs.get('llm_provider_name')}")
    print(f"会话：{thread_id}")
    if fault_policy and fault_policy.enabled:
        print(f"故障注入：{fault_policy.tool_name} 前 {fault_policy.times} 次调用失败")

    graph = await TheWorld(
        InMemorySaver(),
        specs,
        fault_policy=fault_policy,
    )
    recorder = TraceRecorder(
        case_id=args.case_id or "ad-hoc",
        title=args.title or "在线运行",
        mode="online",
        model={
            "provider": specs.get("model_provider"),
            "name": specs.get("llm_provider_name"),
        },
        acceptance_points=args.acceptance or [],
    )
    await _run_turns(graph, args.question, thread_id, recorder)
    _write_trace(recorder, args.trace)
    return 0


# ---------------------------------------------------------------------------
# demo（离线 Mock）
# ---------------------------------------------------------------------------
async def cmd_demo(args: argparse.Namespace) -> int:
    from .offline import get_scenario

    try:
        scenario = get_scenario(args.case)
    except KeyError as exc:
        print(f"[参数错误] {exc}")
        return 2

    print("!" * 72)
    print("离线 Mock 模式：主模型回复是预设脚本，只验证图 / 工具 / Guard 接线，")
    print("不能证明真实模型决策效果。真实效果请看 examples/cases/ 下的在线 Trace。")
    print("!" * 72)

    tools = await get_tools()
    model = scenario.build_model()
    fault_policy = (
        ToolFaultPolicy(scenario.fault_tool, scenario.fault_times)
        if scenario.fault_tool
        else None
    )
    graph = build_graph_with_model(
        main_model=model,
        main_model_with_tools=model,
        main_system_prompt="离线 Mock：主模型回复由脚本预设。",
        tools=tools,
        checkpointer=InMemorySaver(),
        fault_policy=fault_policy,
        hash_check_fn=lambda: False,
        index_ready_fn=lambda: None,
    )
    recorder = TraceRecorder(
        case_id=f"offline-{scenario.key}",
        title=scenario.title,
        mode="offline-mock",
        model={"name": "scripted (offline)"},
        acceptance_points=scenario.acceptance_points,
    )
    await _run_turns(graph, scenario.questions, args.thread_id, recorder)
    _write_trace(recorder, args.trace)
    return 0


# ---------------------------------------------------------------------------
# setup / check / build-index / serve
# ---------------------------------------------------------------------------
def cmd_setup(args: argparse.Namespace) -> int:
    """下载/加载向量模型，并确保索引存在。"""
    print(f"准备向量模型：{EMBEDDING_MODEL_NAME}")
    print("首次运行需要联网下载（默认 HF 镜像 https://hf-mirror.com），已缓存则直接加载。")
    try:
        model = ensure_embedding_model(allow_download=True)
        dimension = embedding_dimension(model)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[失败] {type(exc).__name__}: {exc}")
        print("可尝试：设置 HF_ENDPOINT=https://huggingface.co 后重试，或预先准备 HF 缓存。")
        return 1

    print(f"模型就绪：{EMBEDDING_MODEL_NAME}（维度 {dimension}）")
    status = data_status()
    if not (status.get("sentences_index") and status.get("chunks_index")):
        print("\n索引缺失，开始构建 ...")
        try:
            result = build_index(force=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {type(exc).__name__}: {exc}")
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n环境就绪。下一步：")
    print("  python run.py serve              # 浏览器模式")
    print("  python run.py demo --case normal # 无 API key 的 Mock 模式")
    print("  python run.py ask -q \"...\"       # 真实模型 CLI")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=False)
    except ImportError:  # pragma: no cover
        pass

    try:
        changed = do_we_have_file_change()
    except Exception as exc:  # noqa: BLE001
        changed = f"检查失败：{type(exc).__name__}: {exc}"

    payload = {
        "project_root": str(PROJECT_ROOT),
        "config": public_config(),
        "data": data_status(),
        "document_set_changed": changed,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    try:
        result = build_index(force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"[构建失败] {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import app

    print(f"服务启动：http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="企业知识问答 Agent（RAG + 双 Guard）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="下载/加载向量模型并检查数据索引")
    setup.set_defaults(func=cmd_setup)

    serve = sub.add_parser("serve", help="启动 HTTP API + 浏览器页面")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8010)
    serve.set_defaults(func=cmd_serve)

    ask = sub.add_parser("ask", help="用真实模型跑一轮或多轮问答，并记录 Trace")
    ask.add_argument("-q", "--question", action="append", required=True, help="可重复传入实现多轮")
    ask.add_argument("--thread-id", default=None)
    ask.add_argument("--case-id", default=None)
    ask.add_argument("--title", default=None)
    ask.add_argument("--acceptance", action="append", default=None, help="覆盖的验收点，可重复")
    ask.add_argument("--trace", default=None, help="Trace 输出路径（.json；同名 .md 自动生成）")
    ask.add_argument("--fault-tool", default=None, help="故障注入：让该工具前 N 次调用失败")
    ask.add_argument("--fault-times", type=int, default=1)
    ask.set_defaults(func=cmd_ask, is_async=True)

    demo = sub.add_parser("demo", help="离线 Mock 演示（无需 API key）")
    demo.add_argument(
        "--case",
        required=True,
        choices=["normal", "conflict", "no-support", "paraphrase", "followup", "tool-failure"],
    )
    demo.add_argument("--thread-id", default="offline-demo")
    demo.add_argument("--trace", default=None)
    demo.set_defaults(func=cmd_demo, is_async=True)

    build = sub.add_parser("build-index", help="重建 solid 数据与 FAISS 索引")
    build.add_argument("--force", action="store_true", help="即使文档未变化也强制重建")
    build.set_defaults(func=cmd_build_index)

    check = sub.add_parser("check", help="检查配置、数据、索引与 hash 状态")
    check.set_defaults(func=cmd_check)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "is_async", False):
        return asyncio.run(args.func(args))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
