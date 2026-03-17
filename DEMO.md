# RAGLite Demo Setup

## 1. Clone the repo

```bash
git clone <repo-url>
cd raglite
```

## 2. Start the dev container and switch to the demo branch

```bash
# Open in Docker dev container, then:
git checkout chainlit-demo
```

## 3. Install demo dependencies

```bash
uv sync --extra demo
```

## 4. Configure environment

Copy the `.env` file shared on Slack into the project root.

## 5. Database setup

The demo uses a Neon PostgreSQL database. If the shared connection is alive, skip to step 6.

If the connection is dead (or the account was deactivated):

1. Create a new Neon project at [neon.tech](https://neon.tech) and copy the connection string.
2. Copy the `crag_demo.jsonl` data file shared on Slack into `data/`.
3. Ingest the data:

```bash
uv run python scripts/ingest_crag.py \
    --dataset-path data/crag_demo.jsonl \
    --db-url "postgresql://user:pass@host/dbname" \
    --llm azure/gpt-5-mini \
    --categories music movie sports open
```

## 6. Run the demo

```bash
raglite \
    --db-url "postgresql://user:pass@host/dbname" \
    --llm azure/gpt-5-mini \
    --embedder azure/text-embedding-3-large \
    chainlit
```

The Chainlit UI exposes two toggles in **Settings**:
- **Agentic loop** — enables iterative sub-agent search
- **Self-querying** — uses metadata to filter the search results

## 7. Test questions

Use these to verify the demo is working.

**Set questions**:

> Who are the owners of the Vogtle Electric Generating Plant?

> What are the 4 smallest cities by population in the USA according to the latest census?

**Comparison questions**:

> Which country had a larger GDP in 2023 according to data, China or India? And by how much?

> Which sports organization has a smaller membership, the International Tennis Federation or the International Hockey Federation?
