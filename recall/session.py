"""Short-term (working) memory for one conversation, in Redis.

The last `window` turns are kept verbatim in a list. Older turns are folded into a rolling summary, so a long
conversation keeps its gist at a fixed size. Both keys expire `ttl` seconds after the last turn. The
summariser is pluggable: the default is extractive and offline, and an LLM summariser is an async function with
the same signature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

Summariser = Callable[[str, list[dict]], Awaitable[str]]


async def extractive_summary(previous: str, turns: list[dict], max_lines: int = 20) -> str:
    lines = [l for l in previous.splitlines() if l.strip()]
    for t in turns:
        first = t["text"].strip().split("\n")[0]
        lines.append(f"{t['role']}: {first[:160]}")
    return "\n".join(lines[-max_lines:])


@dataclass
class SessionContext:
    summary: str = ""
    turns: list[dict] = field(default_factory=list)


class SessionMemory:
    def __init__(self, redis, *, window: int = 12, ttl: int = 24 * 3600, summariser: Summariser | None = None):
        self.r, self.window, self.ttl = redis, window, ttl
        self.summarise = summariser or extractive_summary

    @staticmethod
    def _keys(tenant_id: str, session_id: str) -> tuple[str, str]:
        base = f"recall:{tenant_id}:session:{session_id}"
        return base + ":turns", base + ":summary"

    async def add(self, tenant_id: str, session_id: str, role: str, text: str) -> None:
        turns_k, sum_k = self._keys(tenant_id, session_id)
        n = await self.r.rpush(turns_k, json.dumps({"role": role, "text": text}))
        overflow = n - self.window
        if overflow > 0:
            old = [json.loads(x) for x in (await self.r.lpop(turns_k, overflow) or [])]
            prev = await self.r.get(sum_k)
            prev = prev.decode() if isinstance(prev, bytes) else (prev or "")
            await self.r.set(sum_k, await self.summarise(prev, old))
        async with self.r.pipeline(transaction=True) as p:
            p.expire(turns_k, self.ttl)
            p.expire(sum_k, self.ttl)
            await p.execute()

    async def context(self, tenant_id: str, session_id: str) -> SessionContext:
        turns_k, sum_k = self._keys(tenant_id, session_id)
        raw = await self.r.lrange(turns_k, 0, -1)
        s = await self.r.get(sum_k)
        return SessionContext(s.decode() if isinstance(s, bytes) else (s or ""), [json.loads(x) for x in raw])

    async def clear(self, tenant_id: str, session_id: str) -> None:
        await self.r.delete(*self._keys(tenant_id, session_id))
