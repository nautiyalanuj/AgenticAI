"""Build the travel assistant's search-enabled LangGraph workflow.

The graph uses DeepSeek for responses and tool selection, Tavily for current web
search, and a PostgreSQL checkpointer supplied by the caller for conversation
history that survives process restarts.
"""

from __future__ import annotations

# ``os`` lets us read configuration values (such as API keys) from environment
# variables instead of putting secrets directly in the source code.
import os

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# The system prompt is the assistant's standing instruction. We add it every
# time we call the model, so it is still present when an old conversation is
# resumed from PostgreSQL (the checkpoint stores chat messages, not this code).
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
    # client can send requests to it by using DeepSeek's base URL. The API key
    # is read from the environment; the model name and URL have fallback values.
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.2,
    )
    # A LangChain tool is a Python capability the model may ask to use. Binding
    # this tool tells the model what search is available; it does not run a
    # search until the model actually returns a tool call.
    tools = [TavilySearch(max_results=5)]
    model_with_tools = model.bind_tools(tools)

    def assistant(state: MessagesState):
        """Ask DeepSeek for the next response or a Tavily tool call.

        ``MessagesState`` is LangGraph's standard state for a list of chat
        messages. Returning a new ``messages`` list lets LangGraph append the
        model's reply (or tool request) to that state.
        """
        # The ``*`` unpacks the saved message list so the final list is one
        # system message followed by the conversation so far.
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model_with_tools.invoke(messages)]}

    def route_after_assistant(state: MessagesState) -> str:
        """Choose the next graph step based on the model's latest message.

        Tool-enabled chat models put requested actions in ``tool_calls``. If
        that list is empty, the message is a normal answer and the graph ends.
        """
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    # A StateGraph is a workflow whose nodes read and update shared state.
    # This graph loops back after tools, allowing the model to read search
    # results and then produce a user-facing answer. ``END`` marks completion.
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    # The router returns either the "tools" node name or LangGraph's END marker.
    builder.add_conditional_edges("assistant", route_after_assistant)
    builder.add_edge("tools", "assistant")
    return builder.compile(checkpointer=checkpointer)
