from conftest import DSN, run
from recall import MemoryStore
from recall.ingest import ingest

GUIDE = (
    "Crumb & Co brand guide. Our logo is always forest green on cream. Never place it on photos. "
    "Voice: friendly, a little cheeky, short sentences. We never use the word 'artisanal'. "
    "Hero products are the cardamom bun and the 48-hour sourdough. "
    "Posting schedule: Instagram reels on Tuesday and Friday evenings; a WhatsApp offer every Sunday. "
    "Prices: the sourdough is 220 rupees and the cardamom bun is 90 rupees. "
) * 2


def test_llamaindex_pipeline_chunks_embeds_and_stores_documents(tenant):
    async def go():
        s = await MemoryStore.connect(DSN)
        first = await ingest(s, tenant, [("brand-guide.md", GUIDE)], chunk_size=40, chunk_overlap=5)
        again = await ingest(s, tenant, [("brand-guide.md", GUIDE)], chunk_size=40, chunk_overlap=5)
        hits = await s.recall(tenant, "which word must we never use?", kinds=["document"], k=3)
        src = (await (await s.conn.execute("SELECT DISTINCT source FROM memories WHERE tenant_id=%s",
                                           (tenant,))).fetchall())
        await s.close()
        return first, again, hits, src
    first, again, hits, src = run(go())
    assert first["chunks"] >= 3 and first["created"] >= 2
    assert again["created"] == 0 and again["duplicates"] == again["chunks"]     # re-ingesting is idempotent
    assert any("artisanal" in h.content for h in hits)
    assert src == [("brand-guide.md",)]
