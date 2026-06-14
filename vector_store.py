# Embeds text chunks into vectors and stores them in FAISS for semantic retrieval (RAG indexing layer).
import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_index: faiss.IndexFlatL2 | None = None
_chunks: list[str] = []

EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536


def _embed(texts: list[str]) -> np.ndarray:
    client = OpenAI(api_key=os.getenv("API_KEY"))
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def embed_and_store(chunks: list[str]) -> None:
    global _index, _chunks

    vectors = _embed(chunks)

    _index = faiss.IndexFlatL2(DIMENSION)
    _index.add(vectors)
    _chunks = chunks


def retrieve(query: str, k: int = 3) -> list[str]:
    if _index is None or _index.ntotal == 0:
        return []

    vector = _embed([query])
    k = min(k, _index.ntotal)
    _, indices = _index.search(vector, k)

    return [_chunks[i] for i in indices[0] if i < len(_chunks)]
