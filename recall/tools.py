"""Memory tools, written once and registered with each agent framework.

The tenant and user come from the run's dependencies, never from the model's arguments, so a
prompt-injected "remember this for tenant B" cannot cross tenants.
"""
from __future__ import annotations

from dataclasses import dataclass

from .memory import Memory

INSTRUCTIONS = (
    "You are a marketing assistant for one business. You have long-term memory.\n"
    "- Before answering, use what you remember below; call `recall` if you need more.\n"
    "- When the user tells you a durable fact or preference (brand tone, audience, budget, products, what "
    "worked or failed), call `remember` with a short `subject` such as 'brand tone' so a later change replaces it.\n"
    "- Use about_user=true only for this user's personal preferences, not the business's.\n"
    "- If the user asks you to forget something, call `forget` with the memory id shown in brackets.\n"
)


@dataclass
class MemoryDeps:
    memory: Memory
    tenant_id: str
    user_id: str | None = None
    session_id: str | None = None
    last_prompt: str = ""        # set per turn by frameworks whose instructions can't see the prompt


async def remember(deps: MemoryDeps, content: str, kind: str = "fact", subject: str | None = None,
                   importance: float = 0.5, about_user: bool = False) -> str:
    r = await deps.memory.store.remember(deps.tenant_id, content, kind=kind, subject=subject,
                                         user_id=deps.user_id if about_user else None,
                                         importance=max(0.0, min(1.0, importance)))
    if r.action == "duplicate":
        return f"Already known as memory {r.id}; reinforced it."
    if r.action == "superseded":
        return f"Saved as memory {r.id}; it replaces {', '.join(map(str, r.replaced))}."
    return f"Saved as memory {r.id}."


async def recall(deps: MemoryDeps, query: str, k: int = 5) -> str:
    hits = await deps.memory.store.recall(deps.tenant_id, query, user_id=deps.user_id, k=max(1, min(k, 20)))
    if not hits:
        return "Nothing relevant remembered."
    return "\n".join(f"[{h.id}] ({h.kind}) {h.content}" for h in hits)


async def forget(deps: MemoryDeps, memory_id: int) -> str:
    ok = await deps.memory.store.forget(deps.tenant_id, memory_id)
    return f"Forgot memory {memory_id}." if ok else f"No memory {memory_id} for this business."


async def instructions_for(deps: MemoryDeps, prompt: str) -> str:
    block = await deps.memory.context_block(deps.tenant_id, prompt, user_id=deps.user_id, session_id=deps.session_id)
    return INSTRUCTIONS + ("\n" + block if block else "\n(No memories yet.)")
