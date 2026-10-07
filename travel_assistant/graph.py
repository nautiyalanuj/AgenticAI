"""Build the travel assistant's search-enabled LangGraph workflow.

The graph uses DeepSeek for responses and tool selection, Tavily for current web
search, and a PostgreSQL checkpointer supplied by the caller for conversation
history that survives process restarts.
"""

from __future__ import annotations

import os

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# Instructions are sent on each model turn so search behavior remains consistent
# after a conversation is resumed from its saved checkpoint.
SYSTEM_PROMPT = """You are a practical travel research assistant.
Use web search for current travel information, cite the URLs returned by search,
and distinguish confirmed details from estimates. Ask a concise follow-up when
important trip details are missing. Never claim to book or purchase anything."""


def build_graph(checkpointer: PostgresSaver):
    """Create a compiled travel chat graph.

    Args:
        checkpointer: Configured PostgreSQL saver used by LangGraph to persist
            message history for each ``thread_id``.

    Returns:
        A compiled graph ready to invoke with a ``messages`` state and a
        ``configurable.thread_id`` value.
    """
    # DeepSeek exposes an OpenAI-compatible API, so LangChain's ChatOpenAI
    # client can be pointed at DeepSeek's endpoint with its base URL.
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.2,
    )
    # Bind Tavily search so the model can request live sources when needed.
    tools = [TavilySearch(max_results=5)]
    model_with_tools = model.bind_tools(tools)

    def assistant(state: MessagesState):
        """Ask DeepSeek for the next response or a Tavily tool call."""
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model_with_tools.invoke(messages)]}

    def route_after_assistant(state: MessagesState) -> str:
        """Run requested tools, or finish when the model returns a final answer."""
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    # Flow: start -> assistant -> (tools -> assistant)* -> end.
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", route_after_assistant)
    builder.add_edge("tools", "assistant")
    return builder.compile(checkpointer=checkpointer)
