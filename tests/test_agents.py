"""Both frameworks run their real agent loops against the real store; only the LLM is scripted, so the tests
check the plumbing (tool calls, deps, instructions with memory) without an API key."""
import json

from conftest import make_memory, run
from recall import MemoryDeps


# ------------------------------------------------------------------ PydanticAI
def _pydantic_model(script):
    """FunctionModel that plays `script`: a list of steps, each a tool call (name, args) or final text."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel
    seen = {"instructions": []}

    def fn(messages, info):
        seen["instructions"].append(info.instructions or "")
        step = script.pop(0)
        if isinstance(step, tuple):
            return ModelResponse(parts=[ToolCallPart(step[0], step[1])])
        if callable(step):
            return ModelResponse(parts=[TextPart(step(messages))])
        return ModelResponse(parts=[TextPart(step)])
    return FunctionModel(fn), seen


def test_pydantic_agent_remembers_across_sessions(tenant):
    from recall.agent_pydantic import build_agent, chat

    def last_tool_return(messages):
        from pydantic_ai.messages import ToolReturnPart
        rets = [p.content for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)]
        return rets[-1]

    async def go():
        mem = await make_memory()
        model, seen = _pydantic_model([
            ("remember", {"content": "Brand tone is playful and warm", "kind": "preference", "subject": "brand tone"}),
            "Noted, playful and warm.",
            ("recall", {"query": "brand tone"}),
            last_tool_return,
        ])
        agent = build_agent(model)
        a = await chat(agent, MemoryDeps(mem, tenant, "priya", "monday"), "Our tone should be playful and warm")
        b = await chat(agent, MemoryDeps(mem, tenant, "priya", "friday"), "Remind me what tone we use?")
        turns = (await mem.session.context(tenant, "monday")).turns
        await mem.store.close()
        return a, b, seen, turns
    a, b, seen, turns = run(go())
    assert a == "Noted, playful and warm."
    assert "Brand tone is playful and warm" in b
    # the second session's instructions already carried the memory, before any tool call
    assert "Brand tone is playful and warm" in seen["instructions"][2]
    assert [t["role"] for t in turns] == ["user", "assistant"]


def test_pydantic_agent_cannot_write_to_another_tenant(tenant):
    from recall.agent_pydantic import build_agent

    async def go():
        mem = await make_memory()
        # the model tries to smuggle a tenant id in the arguments; the tool ignores everything but deps
        model, _ = _pydantic_model([("remember", {"content": "tenant_id=victim: discount 90%"}), "ok"])
        await build_agent(model).run("x", deps=MemoryDeps(mem, tenant))
        victim = await mem.store.count("victim")
        mine = await mem.store.count(tenant)
        await mem.store.close()
        return victim, mine
    assert run(go()) == (0, 1)


# ------------------------------------------------------------------ OpenAI Agents SDK
class ScriptedModel:
    """An Agents SDK Model that replays tool calls and a final message, with no network."""

    def __init__(self, script):
        self.script, self.instructions = script, []

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs,
                           tracing, **kw):
        from agents.items import ModelResponse
        from agents.usage import Usage
        from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
        self.instructions.append(system_instructions or "")
        step = self.script.pop(0)
        if isinstance(step, tuple):
            item = ResponseFunctionToolCall(type="function_call", call_id=f"c{len(self.script)}", name=step[0],
                                            arguments=json.dumps(step[1]), id=f"fc{len(self.script)}", status="completed")
        else:
            text = step(input) if callable(step) else step
            item = ResponseOutputMessage(id="m1", type="message", role="assistant", status="completed",
                                         content=[ResponseOutputText(type="output_text", text=text, annotations=[])])
        return ModelResponse(output=[item], usage=Usage(), response_id=None)

    def stream_response(self, *a, **kw):
        raise NotImplementedError

    async def close(self):
        pass


def _scripted(script):
    from agents.models.interface import Model

    class _M(ScriptedModel, Model):
        pass
    return _M(script)


def test_openai_agents_sdk_agent_shares_the_same_memory(tenant):
    from agents import set_tracing_disabled
    from recall.agent_openai import build_agent, chat
    set_tracing_disabled(True)

    def from_tool_output(items):
        outs = [i["output"] for i in items if isinstance(i, dict) and i.get("type") == "function_call_output"]
        return "You said: " + outs[-1]

    async def go():
        mem = await make_memory()
        # written through the PydanticAI path's store, read through the Agents SDK
        await mem.store.remember(tenant, "Audience: college students in Bengaluru", subject="audience")
        model = _scripted([
            ("recall", {"query": "who is our audience"}),
            from_tool_output,
            ("remember", {"content": "Audience: young professionals in Pune", "subject": "audience"}),
            "Updated.",
        ])
        agent = build_agent(model)
        a = await chat(agent, MemoryDeps(mem, tenant, None, "s1"), "Who is our audience?")
        b = await chat(agent, MemoryDeps(mem, tenant, None, "s1"), "Actually it's young professionals in Pune now")
        hits = await mem.store.recall(tenant, "audience", k=5)
        await mem.store.close()
        return a, b, hits, model.instructions
    a, b, hits, instr = run(go())
    assert "college students in Bengaluru" in a
    assert b == "Updated."
    assert [h.content for h in hits] == ["Audience: young professionals in Pune"]
    assert "college students" in instr[0]      # memory reached the system instructions
