"""The same memory tools on an OpenAI Agents SDK agent.

    agent = build_agent("gpt-5.1")       # or any Model instance
    result = await Runner.run(agent, "What tone do we use?", context=MemoryDeps(memory, "acme", "priya", "s1"))
"""
from __future__ import annotations

from agents import Agent, RunContextWrapper, Runner, function_tool

from . import tools
from .tools import MemoryDeps


@function_tool
async def remember(ctx: RunContextWrapper[MemoryDeps], content: str, kind: str = "fact", subject: str | None = None,
                   importance: float = 0.5, about_user: bool = False) -> str:
    """Save a durable fact or preference. `subject` (e.g. 'brand tone') lets a later change replace it.
    kind is one of fact, preference, episode."""
    return await tools.remember(ctx.context, content, kind, subject, importance, about_user)


@function_tool
async def recall(ctx: RunContextWrapper[MemoryDeps], query: str, k: int = 5) -> str:
    """Search long-term memory for this business and user."""
    return await tools.recall(ctx.context, query, k)


@function_tool
async def forget(ctx: RunContextWrapper[MemoryDeps], memory_id: int) -> str:
    """Delete a memory by the id shown in brackets."""
    return await tools.forget(ctx.context, memory_id)


def build_agent(model, prompt_for_memory: str = "") -> Agent[MemoryDeps]:
    async def instructions(ctx: RunContextWrapper[MemoryDeps], agent: Agent) -> str:
        return await tools.instructions_for(ctx.context, getattr(ctx.context, "last_prompt", "") or prompt_for_memory)

    return Agent[MemoryDeps](name="recall-openai", instructions=instructions, tools=[remember, recall, forget],
                             model=model)


async def chat(agent: Agent[MemoryDeps], deps: MemoryDeps, text: str) -> str:
    deps.last_prompt = text          # lets the instructions recall memories relevant to this turn
    result = await Runner.run(agent, text, context=deps)
    out = str(result.final_output)
    if deps.session_id:
        await deps.memory.session.add(deps.tenant_id, deps.session_id, "user", text)
        await deps.memory.session.add(deps.tenant_id, deps.session_id, "assistant", out)
    return out
