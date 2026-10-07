"""Run the assistant as a small interactive terminal chat."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from travel_assistant.graph import build_graph


def main() -> None:
    load_dotenv()
    for name in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY"):
        if not os.getenv(name):
            raise SystemExit(f"Missing {name}. Copy .env.example to .env and add your key.")

    parser = argparse.ArgumentParser(description="Chat with the travel research assistant")
    parser.add_argument("--thread-id", default="local", help="Conversation ID to resume (default: local)")
    args = parser.parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("Missing DATABASE_URL in .env")

    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)
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
            result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
            print(f"assistant> {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
