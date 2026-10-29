# Recall: long-term memory for AI agents

An agent that serves a business has to remember things between conversations: the brand's tone, who the
audience is, this month's budget, that one user wants bullet points and another wants paragraphs. It also has
to stop remembering the old tone once it changes, and it must never show one business another business's
facts.

Recall is that memory layer. It is built for a marketing assistant that works for many businesses at once.

| Layer | What it does | Built with |
|---|---|---|
| Long-term memory | Facts, preferences, episodes and document chunks per business and per user. Recall ranks by similarity, recency and importance. | Postgres + **pgvector** (HNSW index) |
| Versioning | A new memory with the same subject supersedes the old one. Near-duplicates are reinforced, not stored twice. Forgetting cascades to old versions. | SQL in one transaction |
| Session memory | The last N turns stay verbatim. Older turns fold into a rolling summary, and both expire. | **Redis** |
| Context engineering | Memories, then the summary, then recent turns, packed into a character budget, weakest first out. | `recall/memory.py` |
| Agents | One set of memory tools (`remember`, `recall`, `forget`) registered on both frameworks. | **PydanticAI** and the **OpenAI Agents SDK** |
| Document ingestion | Brand guides and product sheets are split on sentences, embedded and stored as memories. Re-ingesting is idempotent. | **LlamaIndex** `IngestionPipeline` |

## How it works

```mermaid
flowchart LR
    U([user turn]) --> A[agent: PydanticAI or Agents SDK]
    A -- instructions --> C[context block]
    C --> LT[(Postgres + pgvector<br/>long-term memory)]
    C --> S[(Redis<br/>session turns + summary)]
    A -- remember / recall / forget --> T[memory tools]
    T --> LT
    D[documents] --> L[LlamaIndex pipeline] --> LT
```

**Recall query.** It takes the 40 nearest candidates from the HNSW index inside the caller's scope (this
business, plus this user's private memories), then re-ranks them in SQL:

```
score = 0.70 * cosine similarity + 0.20 * exp(-days since last used / 30) + 0.10 * importance
```

pgvector's iterative index scan keeps reading until enough rows pass the tenant filter, so a small tenant
inside a large table still gets results.

**Tenant safety.** The tenant and user come from the run's dependencies, never from the model's tool
arguments. A prompt-injected "save this for business B" can't cross tenants. There's a test for exactly that.

## Measured

`python eval/run_eval.py` covers three businesses (a bakery, an accounting SaaS and a gym) with overlapping facts:
each has a discount code, a tone, a budget and an audience, so a leak would be easy to hit. There are 26
questions, 4 facts that change mid-history, and 4 users with private preferences. The baseline is a naive
append-only vector memory over the same data.

| Metric | Recall | Naive append-only |
|---|---|---|
| Expected memory in top 3 | 88.5% (23 / 26) | 88.5% |
| Outdated version in top 3 (stale rate) | **0%** | 75% |
| Another business's memory returned | 0 | 0 |
| Another user's private memory returned | 0 | 0 |
| Rows added when the user repeats current facts | **0** | 26 |
| Forgotten memory recalled afterwards | 0 | 0 |
| Recall latency, median | about 4 ms | about 3 ms |

All 3 misses are the same question ("How much can we spend on ads each month?"). The default embedder is
lexical and offline, so "spend ... each month" doesn't match "monthly ad budget". Setting
`RECALL_EMBEDDER=sentence-transformers` swaps in all-MiniLM-L6-v2 at the same 384 dimensions; that run is not
measured here. Latency was measured on a few dozen rows on a laptop. It shows the query plan is sane, not how
it behaves at scale.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Postgres + pgvector, either:
docker compose up -d                                   # pgvector/pgvector:pg17 + redis:7
export RECALL_PG_DSN=postgresql://postgres:recall@127.0.0.1:5432/postgres RECALL_REDIS_URL=redis://127.0.0.1:6379/0
# or, with no Docker: real Postgres 18 + pgvector compiled to WebAssembly (PGlite)
(cd devdb && npm install && node server.mjs)           # serves postgresql://postgres@127.0.0.1:5433/postgres

pytest -q                    # 21 tests
python eval/run_eval.py      # the table above, written to eval/results.json
```

Without `RECALL_REDIS_URL` the tests use fakeredis. CI runs everything against real Postgres + pgvector and
real Redis.

```python
from recall import Memory, MemoryStore, SessionMemory, MemoryDeps
from recall.agent_pydantic import build_agent, chat
import redis.asyncio as redis

memory = Memory(await MemoryStore.connect(), SessionMemory(redis.from_url("redis://localhost")))
agent = build_agent("anthropic:claude-sonnet-5-5")
await chat(agent, MemoryDeps(memory, tenant_id="crumb", user_id="priya", session_id="mon"),
           "Our tone is playful now, emoji are fine")
await chat(agent, MemoryDeps(memory, tenant_id="crumb", user_id="priya", session_id="fri"),
           "Draft an Instagram caption for the cardamom bun")      # remembers the tone in a new session
```

The OpenAI Agents SDK version is the same call with `recall.agent_openai`.

## Tests

There are 21 tests, all against a real Postgres with pgvector:
- **Store:** de-duplication, superseding, tenant and user isolation, a user's subject not overriding the
  business's, recency tie-breaks, forget cascades and per-user erasure, kind filters, usage tracking, and the
  query plan using the HNSW index.
- **Session:** the window and summary fold, TTLs, a custom summariser, and context-block ordering and budget.
- **Agents:** the real PydanticAI and Agents SDK agent loops with a scripted model (no API key). This covers
  remembering in one session and recalling in another, memory reaching the system instructions, both
  frameworks sharing one store, and a model unable to write to another tenant.
- **Ingestion:** LlamaIndex chunking and idempotent re-ingestion.

## Layout

```
recall/store.py           Postgres + pgvector long-term memory
recall/session.py         Redis session memory with a rolling summary
recall/memory.py          context block within a budget
recall/tools.py           framework-neutral memory tools + instructions
recall/agent_pydantic.py  PydanticAI agent
recall/agent_openai.py    OpenAI Agents SDK agent
recall/ingest.py          LlamaIndex ingestion pipeline
recall/embeddings.py      offline hashing embedder; optional sentence-transformers
eval/                     scenarios + eval runner
devdb/                    PGlite server for running without Docker
```
