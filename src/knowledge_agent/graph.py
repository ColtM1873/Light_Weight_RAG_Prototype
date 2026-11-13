"""主图装配：Model A（工具循环）+ Guard 层。

图结构::

    START -> init_turn -> call_main_llm (Model A)
                             |  tool_calls 非空 -> tool_node -> 回 call_main_llm
                             |  tool_calls 为空  -> guard_parse (Model B + 代码校验)
                                                     |  缺字段/引用错 -> feedback -> 回 call_main_llm
                                                     |  pass -> guard_support (Model C)
                                                                 |  unsupported -> feedback -> 回 call_main_llm
                                                                 |  support -> finalize -> 回 call_main_llm
                                                                                (phase=final) -> END

任意节点后 step_count >= MAX_STEPS -> step_limit -> END（明确的未完成状态）。
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph, END, START, StateGraph
from langgraph.types import RetryPolicy

from .config import SYSTEM_PROMPT, load_agent_specs
from .nodes import ToolFaultPolicy, make_guard_nodes, route_after_model_a
from .state import ExtendedMessagesState

# 向后兼容的别名（旧文档/脚本里叫 system_prompt）
system_prompt = SYSTEM_PROMPT


def build_graph_with_model(
    *,
    main_model: Any,
    main_model_with_tools: Any,
    main_system_prompt: str,
    tools: list[Any] | None = None,
    checkpointer: Any = None,
    fault_policy: ToolFaultPolicy | None = None,
    hash_check_fn: Callable[[], bool] | None = None,
    index_ready_fn: Callable[[], Any] | None = None,
) -> CompiledStateGraph:
    """用现成的模型对象装配图（在线 / 离线 Mock 共用）。"""
    tools_by_name = {tool.name: tool for tool in (tools or [])}
    nodes = make_guard_nodes(
        main_model=main_model,
        main_model_with_tools=main_model_with_tools,
        main_system_prompt=main_system_prompt,
        tools_by_name=tools_by_name,
        fault_policy=fault_policy,
        hash_check_fn=hash_check_fn,
        index_ready_fn=index_ready_fn,
    )
    return assemble_graph(nodes, checkpointer)


async def TheWorld(
    thread_checkpoint: Any,
    main_agent_specs: dict[str, Any] | None = None,
    *,
    fault_policy: ToolFaultPolicy | None = None,
) -> CompiledStateGraph:
    """构建主图（真实模型 + RAG 工具）。

    main_agent_specs 需要包含：
        llm_provider_name: 模型名（如 "deepseek-chat"）
        model_api_key:     API key
        model_provider:    （可选）provider 名，如 "deepseek"
        agent_system_prompt:（可选）覆盖默认系统提示
    """
    if main_agent_specs is None:
        main_agent_specs = load_agent_specs()

    setup = getattr(thread_checkpoint, "setup", None)
    if setup is not None:
        await setup()

    init_kwargs: dict[str, Any] = {
        "model": main_agent_specs["llm_provider_name"],
        "api_key": main_agent_specs.get("model_api_key"),
    }
    provider = main_agent_specs.get("model_provider")
    if provider:
        init_kwargs["model_provider"] = provider
    if main_agent_specs.get("base_url"):
        init_kwargs["base_url"] = main_agent_specs["base_url"]
    main_model = init_chat_model(**init_kwargs)
    main_system_prompt = main_agent_specs.get("agent_system_prompt") or SYSTEM_PROMPT

    # RAG 工具：模块本身懒加载，这里只做导入；即使数据尚未生成，
    # 工具调用也会返回明确的失败信息，图仍可运行。
    rag_tools: list[Any] = []
    try:
        from .rag import get_tools

        rag_tools = await get_tools()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"RAG 工具加载失败，将不绑定工具：{type(exc).__name__}: {exc}",
            stacklevel=2,
        )

    main_model_with_tools = main_model.bind_tools(rag_tools) if rag_tools else main_model
    return build_graph_with_model(
        main_model=main_model,
        main_model_with_tools=main_model_with_tools,
        main_system_prompt=main_system_prompt,
        tools=rag_tools,
        checkpointer=thread_checkpoint,
        fault_policy=fault_policy,
    )


def assemble_graph(
    nodes: dict[str, Any], checkpointer: Any = None
) -> CompiledStateGraph:
    """把节点装配成主图（与具体模型解耦，便于测试）。"""
    builder = StateGraph(ExtendedMessagesState)

    builder.add_node("init_turn", nodes["init_turn"])
    builder.add_node(
        "call_main_llm", nodes["call_main_llm"], retry_policy=RetryPolicy(max_attempts=3)
    )
    builder.add_node(
        "tool_node", nodes["tool_node"], retry_policy=RetryPolicy(max_attempts=3)
    )
    builder.add_node("guard_parse", nodes["guard_parse"])
    builder.add_node("guard_support", nodes["guard_support"])
    builder.add_node("finalize", nodes["finalize"])
    builder.add_node("feedback", nodes["feedback"])
    builder.add_node("step_limit", nodes["step_limit"])

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

    return builder.compile(checkpointer=checkpointer)
