# Travel Assistant starter: LangChain + LangGraph

A small Python travel research assistant using:

- **DeepSeek** for chat and tool selection (OpenAI-compatible API)
- **Tavily** for web search
- **LangGraph** for the tool-calling workflow
- **PostgreSQL** for durable conversation checkpoints
- **LangSmith** for tracing and debugging

## Accounts and API keys

Create accounts and copy the API keys from the provider dashboards:

- [LangSmith](https://smith.langchain.com/) — create an API key in workspace settings.
- [Tavily](https://app.tavily.com/) — create an API key in the dashboard.
- DeepSeek key: use the key you already have.

Signup and email verification must be completed by you. Set the keys in `.env` after copying `.env.example`; do not commit `.env`.

## Run locally

Requirements: Python 3.11+ and Docker Compose (or another PostgreSQL 16 instance).

```bash
cp .env.example .env
# Fill in DEEPSEEK_API_KEY, TAVILY_API_KEY and LANGSMITH_API_KEY in .env
docker compose up -d postgres
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m travel_assistant
```

The default thread ID is `local`. Reuse it to continue that conversation; pass `--thread-id` to keep separate conversations. LangSmith tracing is enabled when `LANGSMITH_API_KEY` is configured. PostgreSQL stores LangGraph checkpoints, not application credentials. The Compose database is published on port `5433` so it can coexist with another local PostgreSQL service on `5432`.

## Stop PostgreSQL

```bash
docker compose down
```

The named Docker volume keeps checkpoint data. To remove it, run `docker compose down -v` (this permanently deletes the local database data).
