"""Per-session vector store. Each visitor gets their own index, so one person's
uploaded documents are never searchable by anyone else on the public demo."""
import faiss
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from medrag.config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(EMBED_MODEL)


class SessionStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.chunks = []

    def add(self, new_chunks):
        if not new_chunks:
            return 0
        vecs = np.asarray(get_model().encode(new_chunks), dtype="float32")
        self.index.add(vecs)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, question, k=5):
        if self.index.ntotal == 0:
            return ""
        q = np.asarray(get_model().encode([question]), dtype="float32")
        _, ids = self.index.search(q, min(k, self.index.ntotal))
        return "\n\n".join(self.chunks[i] for i in ids[0] if 0 <= i < len(self.chunks))
