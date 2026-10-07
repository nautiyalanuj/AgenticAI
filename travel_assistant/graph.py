"""LangGraph travel research assistant with durable PostgreSQL checkpoints."""

from __future__ import annotations

import os

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

SYSTEM_PROMPT = """You are a practical travel research assistant.
Use web search for current travel information, cite the URLs returned by search,
and distinguish confirmed details from estimates. Ask a concise follow-up when
important trip details are missing. Never claim to book or purchase anything."""


def build_graph(checkpointer: PostgresSaver):
    """Build and compile the conversational graph using the provided checkpointer."""
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.2,
    )
    tools = [TavilySearch(max_results=5)]
    model_with_tools = model.bind_tools(tools)

    def assistant(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model_with_tools.invoke(messages)]}

    def route_after_assistant(state: MessagesState) -> str:
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", route_after_assistant)
    builder.add_edge("tools", "assistant")
    return builder.compile(checkpointer=checkpointer)
