"""Long-term memory in Postgres + pgvector.

Every memory belongs to a tenant (a business) and optionally to one user inside it. Reads always filter by
tenant, and a user sees tenant-wide memories plus their own, never another user's.

Writes are de-duplicated (a near-identical memory is reinforced, not stored twice) and versioned: a new
memory with the same `subject` supersedes the active one, so "tone: formal" stops being recalled once
"tone: playful" is remembered. Superseded rows stay as history until the newest version is forgotten, which
cascades to the versions it replaced.

Recall takes the nearest candidates from the HNSW index, then re-ranks them in SQL by
similarity, recency of use and importance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import psycopg
from pgvector.psycopg import register_vector_async

from .embeddings import DIM, default_embedder

KINDS = ("fact", "preference", "episode", "document")

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS memories (
  id            bigserial PRIMARY KEY,
  tenant_id     text NOT NULL,
  user_id       text,
  kind          text NOT NULL CHECK (kind IN {KINDS}),
  subject       text,
  content       text NOT NULL,
  source        text,
  importance    real NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
  embedding     vector({DIM}) NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  last_used_at  timestamptz NOT NULL DEFAULT now(),
  use_count     int NOT NULL DEFAULT 0,
  superseded_by bigint REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS memories_scope ON memories (tenant_id, user_id) WHERE superseded_by IS NULL;
CREATE INDEX IF NOT EXISTS memories_subject ON memories (tenant_id, lower(subject)) WHERE superseded_by IS NULL;
"""

# scope: the tenant's shared memories plus this user's own; `%(u)s` NULL means tenant-wide only
_SCOPE = "tenant_id = %(t)s AND (user_id IS NULL OR user_id = %(u)s) AND superseded_by IS NULL"


@dataclass
class Remembered:
    id: int
    action: str                 # created | duplicate | superseded
    replaced: list[int]


@dataclass
class Hit:
    id: int
    content: str
    kind: str
    subject: str | None
    user_id: str | None
    similarity: float
    score: float


class MemoryStore:
    """Async store; one connection, explicit transactions."""

    def __init__(self, conn: psycopg.AsyncConnection, embedder=None, *, dup_threshold: float = 0.92,
                 w_sim: float = 0.70, w_recency: float = 0.20, w_importance: float = 0.10,
                 half_life_days: float = 30.0, candidates: int = 40):
        self.conn, self.embedder = conn, embedder or default_embedder()
        self.dup_threshold, self.candidates, self.half_life = dup_threshold, candidates, half_life_days
        self.w = (w_sim, w_recency, w_importance)

    @classmethod
    async def connect(cls, dsn: str | None = None, **kw) -> "MemoryStore":
        dsn = dsn or os.getenv("RECALL_PG_DSN", "postgresql://postgres@127.0.0.1:5433/postgres")
        # no server-side prepared statements: required behind PgBouncer (transaction mode) and PGlite
        conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True, prepare_threshold=None)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector_async(conn)
        store = cls(conn, **kw)
        await store.migrate()
        return store

    async def migrate(self) -> None:
        for stmt in filter(str.strip, SCHEMA.split(";")):
            await self.conn.execute(stmt)
        try:   # pgvector >= 0.8: keep scanning the index until enough rows pass the tenant filter
            await self.conn.execute("SET hnsw.iterative_scan = relaxed_order")
        except psycopg.Error:
            pass

    async def close(self) -> None:
        await self.conn.close()

    # ------------------------------------------------------------------ writes
    async def remember(self, tenant_id: str, content: str, *, kind: str = "fact", subject: str | None = None,
                       user_id: str | None = None, importance: float = 0.5, source: str | None = None,
                       embedding: np.ndarray | None = None, at: datetime | None = None) -> Remembered:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not content.strip():
            raise ValueError("content is empty")
        emb = np.asarray(embedding if embedding is not None else self.embedder.embed(content), dtype=np.float32)
        at = at or datetime.now(timezone.utc)
        p = dict(t=tenant_id, u=user_id, q=emb, k=kind, s=subject, c=content, src=source,
                 imp=float(importance), at=at)
        async with self.conn.transaction():
            # exact scope for writes: a user's memory never dedups against the tenant's or another user's
            dup = await (await self.conn.execute(
                "SELECT id, 1 - (embedding <=> %(q)s) AS sim FROM memories "
                "WHERE tenant_id = %(t)s AND user_id IS NOT DISTINCT FROM %(u)s AND superseded_by IS NULL "
                "AND kind = %(k)s ORDER BY embedding <=> %(q)s LIMIT 1", p)).fetchone()
            if dup and dup[1] >= self.dup_threshold:
                await self.conn.execute(
                    "UPDATE memories SET importance = GREATEST(importance, %(imp)s), last_used_at = %(at)s, "
                    "use_count = use_count + 1 WHERE id = %(id)s", {**p, "id": dup[0]})
                return Remembered(dup[0], "duplicate", [])
            new_id = (await (await self.conn.execute(
                "INSERT INTO memories (tenant_id, user_id, kind, subject, content, source, importance, embedding, "
                "created_at, last_used_at) VALUES (%(t)s, %(u)s, %(k)s, %(s)s, %(c)s, %(src)s, %(imp)s, %(q)s, "
                "%(at)s, %(at)s) RETURNING id", p)).fetchone())[0]
            replaced = []
            if subject:
                rows = await (await self.conn.execute(
                    "UPDATE memories SET superseded_by = %(new)s WHERE tenant_id = %(t)s "
                    "AND user_id IS NOT DISTINCT FROM %(u)s AND lower(subject) = lower(%(s)s) "
                    "AND superseded_by IS NULL AND id <> %(new)s RETURNING id", {**p, "new": new_id})).fetchall()
                replaced = [r[0] for r in rows]
        return Remembered(new_id, "superseded" if replaced else "created", replaced)

    async def forget(self, tenant_id: str, memory_id: int) -> bool:
        """Delete a memory and, by cascade, the older versions it superseded."""
        cur = await self.conn.execute("DELETE FROM memories WHERE tenant_id = %s AND id = %s", (tenant_id, memory_id))
        return cur.rowcount > 0

    async def forget_user(self, tenant_id: str, user_id: str) -> int:
        cur = await self.conn.execute("DELETE FROM memories WHERE tenant_id = %s AND user_id = %s", (tenant_id, user_id))
        return cur.rowcount

    async def forget_tenant(self, tenant_id: str) -> int:
        cur = await self.conn.execute("DELETE FROM memories WHERE tenant_id = %s", (tenant_id,))
        return cur.rowcount

    # ------------------------------------------------------------------ reads
    async def recall(self, tenant_id: str, query: str, *, user_id: str | None = None, k: int = 5,
                     kinds: list[str] | None = None, min_similarity: float = 0.05, touch: bool = True,
                     now: datetime | None = None) -> list[Hit]:
        q = np.asarray(self.embedder.embed(query), dtype=np.float32)
        ws, wr, wi = self.w
        p = dict(t=tenant_id, u=user_id, q=q, kinds=kinds, n=self.candidates, k=k, now=now or datetime.now(timezone.utc),
                 hl=self.half_life, ws=ws, wr=wr, wi=wi, ms=min_similarity)
        rows = await (await self.conn.execute(f"""
            WITH cand AS (
              SELECT id, content, kind, subject, user_id, importance, last_used_at,
                     1 - (embedding <=> %(q)s) AS sim
              FROM memories
              WHERE {_SCOPE} AND (%(kinds)s::text[] IS NULL OR kind = ANY(%(kinds)s::text[]))
              ORDER BY embedding <=> %(q)s
              LIMIT %(n)s)
            SELECT id, content, kind, subject, user_id, sim,
                   %(ws)s * sim
                 + %(wr)s * exp(-GREATEST(extract(epoch FROM (%(now)s - last_used_at)), 0) / 86400.0 / %(hl)s)
                 + %(wi)s * importance AS score
            FROM cand WHERE sim >= %(ms)s
            ORDER BY score DESC
            LIMIT %(k)s""", p)).fetchall()
        hits = [Hit(r[0], r[1], r[2], r[3], r[4], float(r[5]), float(r[6])) for r in rows]
        if touch and hits:
            await self.conn.execute("UPDATE memories SET last_used_at = %s, use_count = use_count + 1 "
                                    "WHERE id = ANY(%s)", (p["now"], [h.id for h in hits]))
        return hits

    async def history(self, tenant_id: str, subject: str, user_id: str | None = None) -> list[tuple[int, str, bool]]:
        """Every version of a subject, oldest first: (id, content, is_active)."""
        rows = await (await self.conn.execute(
            "SELECT id, content, superseded_by IS NULL FROM memories WHERE tenant_id = %s "
            "AND user_id IS NOT DISTINCT FROM %s AND lower(subject) = lower(%s) ORDER BY id",
            (tenant_id, user_id, subject))).fetchall()
        return [tuple(r) for r in rows]

    async def count(self, tenant_id: str, active_only: bool = True) -> int:
        extra = " AND superseded_by IS NULL" if active_only else ""
        return (await (await self.conn.execute(
            f"SELECT count(*) FROM memories WHERE tenant_id = %s{extra}", (tenant_id,))).fetchone())[0]
