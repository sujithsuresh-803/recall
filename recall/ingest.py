"""Load a business's documents (brand guide, product sheets, past campaigns) into memory with LlamaIndex.

A LlamaIndex IngestionPipeline splits each document on sentence boundaries and embeds the chunks with the same
embedder the store uses. Each chunk becomes a `document` memory for the tenant, so agents recall it alongside
facts.
"""
from __future__ import annotations

from typing import Any

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from pydantic import PrivateAttr

from .store import MemoryStore


class RecallEmbedding(BaseEmbedding):
    """Adapts a Recall embedder to LlamaIndex's embedding interface."""
    _e: Any = PrivateAttr()

    def __init__(self, embedder, **kw):
        super().__init__(model_name=type(embedder).__name__, **kw)
        self._e = embedder

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._e.embed(text).tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)


def pipeline(embedder, chunk_size: int = 120, chunk_overlap: int = 20) -> IngestionPipeline:
    # tokenizer=str.split counts words and avoids downloading a tokenizer
    return IngestionPipeline(transformations=[
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=str.split),
        RecallEmbedding(embedder),
    ])


async def ingest(store: MemoryStore, tenant_id: str, docs: list[tuple[str, str]], **kw) -> dict:
    """docs: (source, text) pairs. Returns counts of chunks created and skipped as duplicates."""
    nodes = await pipeline(store.embedder, **kw).arun(
        documents=[Document(text=t, metadata={"source": s}) for s, t in docs])
    created = dup = 0
    for n in nodes:
        r = await store.remember(tenant_id, n.get_content(), kind="document", source=n.metadata.get("source"),
                                 importance=0.4, embedding=n.embedding)
        created += r.action != "duplicate"
        dup += r.action == "duplicate"
    return {"chunks": len(nodes), "created": created, "duplicates": dup}
