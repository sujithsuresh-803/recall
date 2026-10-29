"""The memory an agent sees: long-term recall plus this conversation, packed into a character budget.

Order matters for context engineering: durable facts about the business and the user come first, then the
summary of earlier turns, then the most recent turns. When the budget runs out, the lowest-ranked memories and
the oldest recent turns are dropped first.
"""
from __future__ import annotations

from dataclasses import dataclass

from .session import SessionMemory
from .store import MemoryStore


@dataclass
class Memory:
    store: MemoryStore
    session: SessionMemory

    async def context_block(self, tenant_id: str, query: str, *, user_id: str | None = None,
                            session_id: str | None = None, k: int = 6, budget_chars: int = 2400) -> str:
        hits = await self.store.recall(tenant_id, query, user_id=user_id, k=k)
        sess = await self.session.context(tenant_id, session_id) if session_id else None
        parts, used = [], 0

        def add(line: str) -> bool:
            nonlocal used
            if used + len(line) + 1 > budget_chars:
                return False
            parts.append(line)
            used += len(line) + 1
            return True

        if hits:
            add("## What you remember about this business and user")
            for h in hits:     # ranked best first, so a full budget drops the weakest
                who = "user" if h.user_id else "business"
                if not add(f"- [{h.id}] ({h.kind}, {who}) {h.content}"):
                    break
        if sess and sess.summary:
            add("## Earlier in this conversation")
            for line in sess.summary.splitlines():
                if not add(line):
                    break
        if sess and sess.turns:
            recent, size = [], 0
            for t in reversed(sess.turns):   # newest first, so the budget keeps the latest turns
                line = f"{t['role']}: {t['text']}"
                if used + size + len(line) + 1 > budget_chars - 30:
                    break
                recent.insert(0, line)
                size += len(line) + 1
            if recent:
                add("## Most recent turns")
                for line in recent:
                    add(line)
        return "\n".join(parts)
