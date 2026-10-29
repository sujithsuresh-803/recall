from datetime import datetime, timedelta, timezone

import pytest

from conftest import DSN, run
from recall import MemoryStore


async def _store(**kw):
    return await MemoryStore.connect(DSN, **kw)


def test_remember_then_recall(tenant):
    async def go():
        s = await _store()
        await s.remember(tenant, "Brand tone is warm and playful, never corporate", kind="preference", subject="brand tone")
        await s.remember(tenant, "Main audience is college students in Bengaluru", subject="audience")
        await s.remember(tenant, "Monthly ad budget is 40,000 rupees", subject="ad budget")
        hits = await s.recall(tenant, "what tone should the posts use?", k=2)
        await s.close()
        return hits
    hits = run(go())
    assert "tone" in hits[0].content.lower()


def test_near_duplicate_is_reinforced_not_stored_twice(tenant):
    async def go():
        s = await _store()
        a = await s.remember(tenant, "We sell sourdough and croissants", importance=0.3)
        b = await s.remember(tenant, "We sell sourdough and croissants.", importance=0.9)
        n = await s.count(tenant)
        imp = (await (await s.conn.execute("SELECT importance FROM memories WHERE id=%s", (a.id,))).fetchone())[0]
        await s.close()
        return a, b, n, imp
    a, b, n, imp = run(go())
    assert b.action == "duplicate" and b.id == a.id and n == 1
    assert imp == pytest.approx(0.9)


def test_same_subject_supersedes_and_old_value_is_not_recalled(tenant):
    async def go():
        s = await _store()
        old = await s.remember(tenant, "Brand tone: formal and corporate", kind="preference", subject="Brand tone")
        new = await s.remember(tenant, "Brand tone: playful, emoji are fine", kind="preference", subject="brand tone")
        hits = await s.recall(tenant, "brand tone", k=5)
        hist = await s.history(tenant, "brand tone")
        await s.close()
        return old, new, hits, hist
    old, new, hits, hist = run(go())
    assert new.action == "superseded" and new.replaced == [old.id]
    assert [h.id for h in hits] == [new.id]
    assert hist == [(old.id, "Brand tone: formal and corporate", False), (new.id, "Brand tone: playful, emoji are fine", True)]


def test_tenants_never_see_each_other(tenant):
    other = tenant + "-other"

    async def go():
        s = await _store()
        await s.remember(tenant, "Launch discount code is CRUMB20", subject="discount code")
        await s.remember(other, "Launch discount code is LEDGER50", subject="discount code")
        mine = await s.recall(tenant, "launch discount code", k=10)
        theirs = await s.recall(other, "launch discount code", k=10)
        await s.close()
        return mine, theirs
    mine, theirs = run(go())
    assert [h.content for h in mine] == ["Launch discount code is CRUMB20"]
    assert [h.content for h in theirs] == ["Launch discount code is LEDGER50"]


def test_user_memories_are_private_to_that_user(tenant):
    async def go():
        s = await _store()
        await s.remember(tenant, "Company posts go out at 9am", subject="posting time")
        await s.remember(tenant, "Priya wants drafts as bullet points", user_id="priya", subject="draft format")
        await s.remember(tenant, "Arjun wants drafts as full paragraphs", user_id="arjun", subject="draft format")
        priya = await s.recall(tenant, "draft format posts", user_id="priya", k=10)
        arjun = await s.recall(tenant, "draft format posts", user_id="arjun", k=10)
        shared = await s.recall(tenant, "draft format posts", k=10)
        await s.close()
        return priya, arjun, shared
    priya, arjun, shared = run(go())
    assert {h.user_id for h in priya} == {None, "priya"}
    assert {h.user_id for h in arjun} == {None, "arjun"}
    assert {h.user_id for h in shared} == {None}


def test_user_subject_does_not_supersede_business_subject(tenant):
    async def go():
        s = await _store()
        biz = await s.remember(tenant, "Posting time for the brand is 9am", subject="posting time")
        mine = await s.remember(tenant, "Remind me at 6pm, my own posting time", user_id="priya", subject="posting time")
        hist = await s.history(tenant, "posting time")
        await s.close()
        return biz, mine, hist
    biz, mine, hist = run(go())
    assert mine.action == "created"
    assert hist == [(biz.id, "Posting time for the brand is 9am", True)]


def test_recency_breaks_ties_between_equally_similar_memories(tenant):
    now = datetime.now(timezone.utc)

    async def go():
        s = await _store()
        old = await s.remember(tenant, "Campaign idea: monsoon chai combo offer", at=now - timedelta(days=120))
        new = await s.remember(tenant, "Campaign idea: monsoon combo offer for chai", at=now - timedelta(days=1))
        hits = await s.recall(tenant, "monsoon chai combo campaign idea", k=2, now=now, touch=False)
        await s.close()
        return old, new, hits
    old, new, hits = run(go())
    assert [h.id for h in hits] == [new.id, old.id]


def test_forget_cascades_to_older_versions(tenant):
    async def go():
        s = await _store()
        await s.remember(tenant, "Budget is 20k a month", subject="budget")
        v2 = await s.remember(tenant, "Budget is 35k a month", subject="budget")
        ok = await s.forget(tenant, v2.id)
        left = await s.count(tenant, active_only=False)
        wrong_tenant = await s.forget(tenant + "-x", v2.id)
        await s.close()
        return ok, left, wrong_tenant
    ok, left, wrong_tenant = run(go())
    assert ok and left == 0 and not wrong_tenant


def test_forget_user_erases_only_that_user(tenant):
    async def go():
        s = await _store()
        await s.remember(tenant, "Shared: we are a bakery")
        await s.remember(tenant, "Priya likes short captions", user_id="priya")
        await s.remember(tenant, "Priya is vegetarian", user_id="priya")
        n = await s.forget_user(tenant, "priya")
        left = await s.count(tenant)
        await s.close()
        return n, left
    assert run(go()) == (2, 1)


def test_kind_filter_and_validation(tenant):
    async def go():
        s = await _store()
        await s.remember(tenant, "Brand guide: logo is always green", kind="document")
        await s.remember(tenant, "Logo was redesigned in March", kind="episode")
        docs = await s.recall(tenant, "logo", kinds=["document"], k=5)
        with pytest.raises(ValueError):
            await s.remember(tenant, "x", kind="gossip")
        with pytest.raises(ValueError):
            await s.remember(tenant, "   ")
        await s.close()
        return docs
    docs = run(go())
    assert [d.kind for d in docs] == ["document"]


def test_recall_marks_memories_as_used(tenant):
    async def go():
        s = await _store()
        m = await s.remember(tenant, "Best performing post format is short reels")
        await s.recall(tenant, "best post format", k=1)
        await s.recall(tenant, "best post format", k=1)
        uses = (await (await s.conn.execute("SELECT use_count FROM memories WHERE id=%s", (m.id,))).fetchone())[0]
        await s.close()
        return uses
    assert run(go()) == 2


def test_recall_uses_the_hnsw_index():
    async def go():
        s = await _store()
        await s.conn.execute("SET enable_seqscan = off")
        q = s.embedder.embed("anything")
        plan = await (await s.conn.execute(
            "EXPLAIN SELECT id FROM memories ORDER BY embedding <=> %s LIMIT 5", (q,))).fetchall()
        await s.conn.execute("RESET enable_seqscan")
        await s.close()
        return " ".join(r[0] for r in plan)
    assert "memories_embedding_hnsw" in run(go())
