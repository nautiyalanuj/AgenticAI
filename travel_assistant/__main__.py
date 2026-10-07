"""Run the assistant as an interactive terminal chat.

Configuration is loaded from ``.env``. LangGraph checkpoints are stored in
PostgreSQL, while LangSmith tracing is enabled by the LangChain environment
variables documented in the project README.
"""

from __future__ import annotations

# ``argparse`` handles command-line options such as ``--thread-id``.
import argparse
# ``os.getenv`` reads optional settings from the process environment.
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from travel_assistant.graph import build_graph


def main() -> None:
    """Validate configuration, open PostgreSQL, and serve chat turns."""
    # Load key/value settings from the local .env file into environment
    # variables. This is convenient for development and keeps secrets out of git.
    load_dotenv()
    # Fail early with a useful message instead of waiting for an API request to
    # fail later with a harder-to-understand authentication error.
    for name in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY"):
        if not os.getenv(name):
            raise SystemExit(f"Missing {name}. Copy .env.example to .env and add your key.")

    parser = argparse.ArgumentParser(description="Chat with the travel research assistant")
    # A thread ID is the key LangGraph uses to find one conversation's saved
    # state. The default makes a simple local run easy to start and resume.
    parser.add_argument("--thread-id", default="local", help="Conversation ID to resume (default: local)")
    args = parser.parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("Missing DATABASE_URL in .env")

    # ``with`` is a context manager: it opens the PostgreSQL connection here
    # and closes it automatically when the indented block finishes, even if an
    # error occurs. The saver is LangGraph's interface to checkpoint storage.
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        # Create LangGraph's checkpoint tables if this is the first run.
        checkpointer.setup()
        graph = build_graph(checkpointer)
        # LangGraph reads ``configurable.thread_id`` to load and save the right
        # conversation history on every invoke call.
        config = {"configurable": {"thread_id": args.thread_id}}
        print(f"Travel assistant ready (thread: {args.thread_id}). Type 'exit' to quit.")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in {"exit", "quit"}:
                break
            if not text:
                continue
            # HumanMessage marks this input as coming from the user. ``invoke``
            # runs the graph once; the checkpointer restores prior messages,
            # adds this turn, and saves the updated conversation.
            result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
            # The graph returns its final state; the last message is the reply
            # after any tool calls and follow-up model turn have completed.
            print(f"assistant> {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
