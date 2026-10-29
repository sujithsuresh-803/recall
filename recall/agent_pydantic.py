"""A PydanticAI agent with long-term memory.

    agent = build_agent("anthropic:claude-sonnet-5-5")
    out = await agent.run("Our tone is playful now", deps=MemoryDeps(memory, "acme", "priya", "s1"))
"""
from __future__ import annotations

from pydantic_ai import Agent, RunContext

from . import tools
from .tools import MemoryDeps


def build_agent(model) -> Agent[MemoryDeps, str]:
    agent = Agent(model, deps_type=MemoryDeps, name="recall-pydantic")

    @agent.instructions
    async def with_memory(ctx: RunContext[MemoryDeps]) -> str:
        prompt = ctx.prompt if isinstance(ctx.prompt, str) else ""
        return await tools.instructions_for(ctx.deps, prompt)

    @agent.tool
    async def remember(ctx: RunContext[MemoryDeps], content: str, kind: str = "fact", subject: str | None = None,
                       importance: float = 0.5, about_user: bool = False) -> str:
        """Save a durable fact or preference. `subject` (e.g. 'brand tone') lets a later change replace it.
        kind is one of fact, preference, episode."""
        return await tools.remember(ctx.deps, content, kind, subject, importance, about_user)

    @agent.tool
    async def recall(ctx: RunContext[MemoryDeps], query: str, k: int = 5) -> str:
        """Search long-term memory for this business and user."""
        return await tools.recall(ctx.deps, query, k)

    @agent.tool
    async def forget(ctx: RunContext[MemoryDeps], memory_id: int) -> str:
        """Delete a memory by the id shown in brackets."""
        return await tools.forget(ctx.deps, memory_id)

    return agent


async def chat(agent: Agent[MemoryDeps, str], deps: MemoryDeps, text: str) -> str:
    """One turn: run the agent, then log both sides to session memory."""
    out = await agent.run(text, deps=deps)
    if deps.session_id:
        await deps.memory.session.add(deps.tenant_id, deps.session_id, "user", text)
        await deps.memory.session.add(deps.tenant_id, deps.session_id, "assistant", out.output)
    return out.output
