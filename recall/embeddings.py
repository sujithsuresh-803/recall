"""Text embeddings.

The default `HashingEmbedder` is deterministic and offline: hashed unigrams and bigrams, signed and
L2-normalised. It is lexical, so it matches shared words rather than meaning; that keeps tests and the
eval reproducible with no model download. Set RECALL_EMBEDDER=sentence-transformers to use
all-MiniLM-L6-v2 (same 384 dimensions) for real semantic recall.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

DIM = 384
_STOP = set("a an the and or of to in on for with is are was were be been it its this that these those our "
            "we you your they their i my me at by as from what which who how do does did should can will "
            "would could has have had about any all".split())
_WORD = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


class HashingEmbedder:
    dim = DIM

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokens(text)
        feats = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = hashlib.blake2b(f.encode(), digest_size=8).digest()
            v[int.from_bytes(h[:4], "little") % self.dim] += 1.0 if h[4] & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


class SentenceTransformerEmbedder:
    dim = DIM

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer   # optional dependency
        self._m = SentenceTransformer(model)

    def embed(self, text: str) -> np.ndarray:
        return self._m.encode(text, normalize_embeddings=True).astype(np.float32)

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        return list(self._m.encode(texts, normalize_embeddings=True).astype(np.float32))


def default_embedder():
    if os.getenv("RECALL_EMBEDDER") == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return HashingEmbedder()
