"""主图：Model A（工具循环）+ Guard 层。

图结构（详见 guard_plan.md）::

    START -> init_turn -> call_main_llm (Model A)
                             |  tool_calls 非空 -> tool_node -> 回 call_main_llm
                             |  tool_calls 为空  -> guard_parse (Model B + 代码校验)
                                                     |  缺字段/引用错 -> feedback -> 回 call_main_llm
                                                     |  pass -> guard_support (Model C)
                                                                 |  unsupported -> feedback -> 回 call_main_llm
                                                                 |  support 且无反馈 -> END
                                                                 |  support 且有反馈 -> finalize -> 回 call_main_llm
                                                                                        (phase=final)
                                                                                        -> END（不再校验）

范围：只实现规划中的 Model A + Guard 链路。
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph, END, START, StateGraph
from langgraph.types import RetryPolicy

from guard_nodes import make_guard_nodes, route_after_model_a
from variant_message_state import ExtendedMessagesState

system_prompt = """
你是一个企业知识问答小助手。你需要借助文档检索等工具，帮助员工找到制度或产品问题的可靠答案。
必须要文档完全支持才可以给出明确答复，不能靠自己臆测。
要注意文档中的细微措辞要求，以及提及的使用范围和条件。如果文档之间明确说明有优先级还有考虑优先级覆盖问题。
如果没有文档资料完全地，明确地支持回答，就不要给出明确的答复，而是只给出相关信息。
如果多个文档内容互相冲突，在没有完全明确的规定指出某个文档可以override其他文档时，应当列出冲突，让用户再进行确认，不要给出答案。
最终，你需要答复用户你的自然语言回复，包括文档明确支持时的回复和文档不明确支持时的解释等等。
以及结构化的状态，包括：引用文档的ID以及相应的检索返回内容（原文片段），资料的缺口，以及任务的状态（是否完成，是否文档冲突，是否无文档可以支撑）
注意：引用只给出文档ID和原文片段即可，片段ID、片段类型等属于内部实现细节，不要出现在给用户的回复里。
"""


ENV_FILE = Path(__file__).resolve().parent / ".env"


def load_agent_specs() -> dict[str, Any]:
    """从项目根目录的 .env（或环境变量）读取模型配置。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(ENV_FILE)

    provider = (os.getenv("LLM_PROVIDER") or "deepseek").strip()
    model = (os.getenv("LLM_MODEL") or "deepseek-chat").strip()
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not api_key:
        api_key = (os.getenv(f"{provider.upper()}_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(f"没有找到 API key：请在 {ENV_FILE} 里填写 LLM_API_KEY=...")

    specs: dict[str, Any] = {
        "llm_provider_name": model,
        "model_provider": provider,
        "model_api_key": api_key,
    }
    base_url = (os.getenv("LLM_BASE_URL") or "").strip()
    if base_url:
        specs["base_url"] = base_url
    system_prompt_override = os.getenv("AGENT_SYSTEM_PROMPT")
    if system_prompt_override:
        specs["agent_system_prompt"] = system_prompt_override
    return specs


async def TheWorld(
    thread_checkpoint: Any,
    main_agent_specs: dict[str, Any] | None = None,
) -> CompiledStateGraph:
    """构建主图。

    main_agent_specs 需要包含：
        llm_provider_name: 模型名（如 "deepseek-chat" 或 "deepseek:deepseek-chat"）
        model_api_key:     API key
        model_provider:    （可选）provider 名，如 "deepseek"
        agent_system_prompt:（可选）覆盖默认 system_prompt
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
    main_system_prompt = main_agent_specs.get("agent_system_prompt") or system_prompt

    # RAG 工具：data/ 与索引尚未生成时降级为空工具列表，主图仍可运行。
    rag_tools: list[Any] = []
    try:
        from query_it_up import get_tools

        rag_tools = await get_tools()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"RAG 工具加载失败，将不绑定工具：{type(exc).__name__}: {exc}",
            stacklevel=2,
        )

    tools_by_name = {tool.name: tool for tool in rag_tools}
    main_model_with_tools = main_model.bind_tools(rag_tools) if rag_tools else main_model

    nodes = make_guard_nodes(
        main_model=main_model,
        main_model_with_tools=main_model_with_tools,
        main_system_prompt=main_system_prompt,
        tools_by_name=tools_by_name,
    )

    return assemble_graph(nodes, thread_checkpoint)


def assemble_graph(
    nodes: dict[str, Any], checkpointer: Any = None
) -> CompiledStateGraph:
    """把 Guard 节点装配成主图（与具体模型解耦，便于测试）。"""
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
