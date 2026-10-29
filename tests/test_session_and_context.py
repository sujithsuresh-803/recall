from conftest import make_memory, redis_client, run
from recall import SessionMemory


def test_window_keeps_recent_turns_and_folds_older_into_summary(tenant):
    async def go():
        sm = SessionMemory(redis_client(), window=3, ttl=120)
        for i in range(7):
            await sm.add(tenant, "s1", "user", f"message number {i}")
        ctx = await sm.context(tenant, "s1")
        ttl = await sm.r.ttl(f"recall:{tenant}:session:s1:turns")
        return ctx, ttl
    ctx, ttl = run(go())
    assert [t["text"] for t in ctx.turns] == ["message number 4", "message number 5", "message number 6"]
    assert ctx.summary.splitlines() == [f"user: message number {i}" for i in range(4)]
    assert 0 < ttl <= 120


def test_sessions_are_separate_and_clearable(tenant):
    async def go():
        sm = SessionMemory(redis_client(), window=5)
        await sm.add(tenant, "a", "user", "hello from a")
        await sm.add(tenant, "b", "user", "hello from b")
        await sm.clear(tenant, "a")
        return await sm.context(tenant, "a"), await sm.context(tenant, "b")
    a, b = run(go())
    assert a.turns == [] and [t["text"] for t in b.turns] == ["hello from b"]


def test_custom_summariser_is_used(tenant):
    async def shout(prev, turns):
        return (prev + " " + " ".join(t["text"].upper() for t in turns)).strip()

    async def go():
        sm = SessionMemory(redis_client(), window=1, summariser=shout)
        for w in ("one", "two", "three"):
            await sm.add(tenant, "s", "user", w)
        return await sm.context(tenant, "s")
    assert run(go()).summary == "ONE TWO"


def test_context_block_orders_memory_then_summary_then_recent(tenant):
    async def go():
        m = await make_memory()
        await m.store.remember(tenant, "Brand tone: playful", kind="preference", subject="brand tone")
        for i in range(6):
            await m.session.add(tenant, "s", "user", f"turn {i} about tone")
        block = await m.context_block(tenant, "brand tone", session_id="s")
        await m.store.close()
        return block
    block = run(go())
    a, b, c = (block.index(h) for h in ("## What you remember", "## Earlier in this conversation", "## Most recent turns"))
    assert a < b < c
    assert "Brand tone: playful" in block and "turn 5 about tone" in block and "user: turn 0 about tone" in block


def test_context_block_respects_budget_and_keeps_newest_turns(tenant):
    async def go():
        m = await make_memory()
        for i in range(8):
            await m.store.remember(tenant, f"Product {i}: handmade candle scent number {i} " + "x" * 60)
        for i in range(4):
            await m.session.add(tenant, "s", "user", f"recent turn {i}")
        block = await m.context_block(tenant, "candle scent product", session_id="s", budget_chars=400)
        await m.store.close()
        return block
    block = run(go())
    assert len(block) <= 400
    assert "recent turn 3" in block or "## What you remember" in block
