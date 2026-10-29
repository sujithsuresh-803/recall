"""Tests run against a real Postgres + pgvector (RECALL_PG_DSN; locally `node devdb/server.mjs`) and Redis
(RECALL_REDIS_URL if set, else fakeredis). Each test uses fresh tenant ids, so tests never share data."""
import asyncio
import os
import sys
import uuid

import pytest

DSN = os.getenv("RECALL_PG_DSN", "postgresql://postgres@127.0.0.1:5433/postgres")


def run(coro):
    # psycopg's async driver needs a selector event loop on Windows
    if sys.platform == "win32":
        return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(coro)


@pytest.fixture(scope="session", autouse=True)
def _pg_available():
    import psycopg
    try:
        psycopg.connect(DSN, connect_timeout=3).close()
    except Exception as e:
        pytest.exit(f"Postgres + pgvector not reachable at {DSN}: {e}. Start it with `node devdb/server.mjs`.", 2)


@pytest.fixture
def tenant():
    return "t-" + uuid.uuid4().hex[:10]


def redis_client():
    url = os.getenv("RECALL_REDIS_URL")
    if url:
        import redis.asyncio as r
        return r.from_url(url)
    import fakeredis
    return fakeredis.FakeAsyncRedis()


async def make_memory(**store_kw):
    from recall import Memory, MemoryStore, SessionMemory
    store = await MemoryStore.connect(DSN, **store_kw)
    return Memory(store, SessionMemory(redis_client(), window=4, ttl=600))
