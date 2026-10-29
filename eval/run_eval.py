"""Memory eval: three businesses with overlapping facts, updates and per-user preferences.

Measures, for Recall and for a naive append-only vector memory over the same data:
  recall@3          the expected memory is in the top 3 for a question
  stale rate        an outdated version (since replaced) appears in the top 3
  cross-tenant hits a memory from another business appears at all
  cross-user hits   another user's private memory appears at all
plus duplicate handling on re-ingest, forgetting, and recall latency.

    node devdb/server.mjs          # or RECALL_PG_DSN=postgresql://... for any Postgres with pgvector
    python eval/run_eval.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recall import MemoryStore  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


async def load(store: MemoryStore, tenant: str, sc: dict, naive: bool) -> dict:
    ids = {}
    for m in sc["memories"]:   # in time order; later memories with the same subject are updates
        r = await store.remember(tenant, m["text"], kind=m.get("kind", "fact"), user_id=m.get("user"),
                                 subject=None if naive else m.get("subject"), importance=m.get("importance", 0.5))
        ids[m["key"]] = r.id
    return ids


async def evaluate(naive: bool, run_id: str) -> dict:
    kw = dict(dup_threshold=1.01, w_sim=1.0, w_recency=0.0, w_importance=0.0) if naive else {}
    store = await MemoryStore.connect(**kw)
    tenants = {name: f"eval-{run_id}-{'naive' if naive else 'recall'}-{name}" for name in SCENARIOS}
    ids, owner = {}, {}
    for name, sc in SCENARIOS.items():
        for key, mid in (await load(store, tenants[name], sc, naive)).items():
            ids[(name, key)] = mid
            owner[mid] = (name, next(m.get("user") for m in sc["memories"] if m["key"] == key))
    stale_keys = {(name, m["key"]) for name, sc in SCENARIOS.items() for m in sc["memories"] if m.get("stale")}

    q_total = q_hit = stale_q = stale_hits = x_tenant = x_user = 0
    lat = []
    misses = []
    for name, sc in SCENARIOS.items():
        for q in sc["questions"]:
            t0 = time.perf_counter()
            hits = await store.recall(tenants[name], q["q"], user_id=q.get("user"), k=3, touch=False)
            lat.append((time.perf_counter() - t0) * 1000)
            got = [h.id for h in hits]
            q_total += 1
            if ids[(name, q["expect"])] in got:
                q_hit += 1
            else:
                misses.append(f"{name}: {q['q']}")
            if q.get("stale"):
                stale_q += 1
                stale_hits += any(ids[(name, s)] in got for s in q["stale"])
            for h in hits:
                b, u = owner.get(h.id, ("?", None))
                x_tenant += b != name
                x_user += u is not None and u != q.get("user")

    # the user repeats every current fact: Recall should reinforce, not duplicate
    before = sum([await store.count(t, active_only=False) for t in tenants.values()])
    for name, sc in SCENARIOS.items():
        await load(store, tenants[name], {"memories": [m for m in sc["memories"] if not m.get("stale")]}, naive)
    after = sum([await store.count(t, active_only=False) for t in tenants.values()])

    # forget one memory per business and ask for it again
    forgotten_seen = 0
    for name, sc in SCENARIOS.items():
        f = sc["forget"]
        await store.forget(tenants[name], ids[(name, f["key"])])
        hits = await store.recall(tenants[name], f["q"], k=5, touch=False)
        forgotten_seen += any(h.id == ids[(name, f["key"])] for h in hits)

    for t in tenants.values():
        await store.forget_tenant(t)
    await store.close()
    return {
        "questions": q_total,
        "recall_at_3": round(q_hit / q_total, 3),
        "stale_rate": round(stale_hits / stale_q, 3) if stale_q else 0.0,
        "stale_questions": stale_q,
        "cross_tenant_hits": x_tenant,
        "cross_user_hits": x_user,
        "rows_added_on_reingest": after - before,
        "forgotten_still_recalled": forgotten_seen,
        "recall_ms_p50": round(statistics.median(lat), 1),
        "misses": misses,
    }


async def main():
    run_id = uuid.uuid4().hex[:6]
    res = {"recall": await evaluate(False, run_id), "naive_append_only": await evaluate(True, run_id)}
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(res, indent=2))
    rows = ["questions", "recall_at_3", "stale_rate", "cross_tenant_hits", "cross_user_hits",
            "rows_added_on_reingest", "forgotten_still_recalled", "recall_ms_p50"]
    print(f"{'metric':28}{'Recall':>10}{'naive':>10}")
    for r in rows:
        print(f"{r:28}{res['recall'][r]!s:>10}{res['naive_append_only'][r]!s:>10}")
    for m in res["recall"]["misses"]:
        print("  miss:", m)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
    else:
        asyncio.run(main())
