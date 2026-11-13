import uuid
import langchain_core
from typing import Any
from langchain.messages import RemoveMessage
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage , AIMessage
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
from langgraph.types import interrupt , Command
from langchain_core.tools.structured import StructuredTool
from langchain_huggingface import HuggingFaceEmbeddings
from variant_message_state import ExtendedMessagesState
from various_prompts import summery_promt,ReAct_system_prompt
from sub_agent_sub_graph_factory_new import build_sub_agent
from langgraph.types import RetryPolicy
from langgraph.graph.state import CompiledStateGraph,StateGraph,START,END


"""  --- Under Construction ---
from query_it_up import get_tools

async def TheWorld(
                thread_checkpoint,
                main_agent_specs : dict[str , Any],
                ) -> CompiledStateGraph:

    await thread_checkpoint.setup()


    main_model = init_chat_model(model = main_agent_specs["llm_provider_name"] , 
                                    api_key = main_agent_specs["model_api_key"])
    main_system_prompt = main_agent_specs["agent_system_prompt"]
    rag_tools = await get_tools()
    
#______________________________________ main_graph _____________________________________________________________________________
#______________________________________ main_graph _____________________________________________________________________________
#______________________________________ memory_tools _____________________________________________________________________________

    main_model_with_tools = main_model.bind_tools(rag_tools)

#______________________________________ main_graph ______________________________________________________________________
    main_agent_builder = StateGraph(ExtendedMessagesState)


#______________________________________ main_graph _______________________________________________________________________________________________________________
    async def we_dont_want_unresponded_human_msg (state):
        msg_list = state["messages"]
        len_msg_list = len(msg_list)
        if len_msg_list >= 2:
            last_second_msg = msg_list[-2]
            if isinstance(last_second_msg , HumanMessage):
                return Command(goto = "we_dont_want_unresponded_human_msg",
                               update= {  "messages" : [RemoveMessage ( id = last_second_msg.id)] })
            else:
                return Command(goto= "extract_human_message")
        else:
            return Command(goto= "extract_human_message")
    (
        main_agent_builder
        .add_node("we_dont_want_unresponded_human_msg" , we_dont_want_unresponded_human_msg)
        .add_edge(START , "we_dont_want_unresponded_human_msg")
    )

    async def call_main_llm (state: ExtendedMessagesState ):
        msg_list = state.get("messages", [])
        feed_in_msg_list = (
            [SystemMessage(content= main_system_prompt )]
            + msg_list
            )
        
        response = await main_model_with_tools.ainvoke(feed_in_msg_list )
        return {"messages" : [response] }
    
    async def route_to_func( state : ExtendedMessagesState):
        node_list = []
        tool_node_bool = False
        for agent_call in state["messages"][-1].tool_calls:
    
    async def tool_node_front(state: ExtendedMessagesState):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_name = tool_call["name"]
            if tool_name not in main_nonagent_tools_by_name:
                result.append(ToolMessage(
                    content="tool call invalid, tools could be dynamically attached or removed, please make sure this tool still remains in your tool_list",
                    tool_call_id=tool_call["id"],
                    name = tool_name
                    ))
                continue
            tool = main_nonagent_tools_by_name[tool_name]
            try:
                observation = await tool.ainvoke(tool_call["args"])
                result.append(ToolMessage(
                    content=observation,
                    tool_call_id=tool_call["id"],
                    name = tool_name
                    ))
            except Exception as e:
                result.append(ToolMessage(
                    content=str(e),
                    tool_call_id=tool_call["id"],
                    name = tool_name
                    ))
        return {"messages": result}

    async def tool_node(state: ExtendedMessagesState):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_name = tool_call["name"]
            if tool_name not in main_nonagent_tools_by_name:
                result.append(ToolMessage(
                    content="You are not supposed to call this tool when depicting HTML report.",
                    tool_call_id=tool_call["id"],
                    name = tool_name
                    ))
                continue
            tool = main_nonagent_tools_by_name[tool_name]
            try:
                observation = await tool.ainvoke(tool_call["args"])
                result.append(ToolMessage(
                    content=observation,
                    tool_call_id=tool_call["id"],
                    name = tool_name
                    ))
            except Exception as e:
                result.append(ToolMessage(
                    content=str(e),
                    tool_call_id=tool_call["id"],
                    name = tool_name
                    ))
        return {"messages": result}

    async def should_continue_main(state:ExtendedMessagesState):
        if state["messages"][-1].tool_calls:
            return "tool_node"
        else:
            return END
    
    list_of_str_subgraphs = [tool.name for tool in sub_agent_tools]

    main_agent_builder.add_node(query_to_produce_html)
    main_agent_builder.add_node(produce_html_call_llm , retry_policy= RetryPolicy(max_attempts=3))
    main_agent_builder.add_node(tool_node , retry_policy= RetryPolicy(max_attempts=3))

    main_agent_builder.add_node(delegate_instructions)
    main_agent_builder.add_node(consume_submitted_reports)
    main_agent_builder.add_node(tool_node_front , retry_policy= RetryPolicy(max_attempts=3))
    for name,subgraph in sub_agent_dict.items():
        main_agent_builder.add_node(name,subgraph , retry_policy= RetryPolicy(max_attempts=3))
    main_agent_builder.add_edge("call_main_llm" , "delegate_instructions")
    main_agent_builder.add_conditional_edges("delegate_instructions" , route_to_func )
    for node in list_of_str_subgraphs:
        main_agent_builder.add_edge(node , "consume_submitted_reports")
    main_agent_builder.add_edge("tool_node_front","consume_submitted_reports")
    main_agent_builder.add_edge("consume_submitted_reports","call_main_llm")

    main_agent_builder.add_conditional_edges("produce_html_call_llm" , should_continue_main )
    main_agent_builder.add_edge("tool_node","produce_html_call_llm")

    main_agent = main_agent_builder.compile(checkpointer= thread_checkpoint , store = store)

    return main_agent

"""