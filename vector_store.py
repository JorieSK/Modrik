# Embeds text chunks into vectors and stores them in FAISS for semantic retrieval (RAG indexing layer).
import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_index: faiss.IndexFlatIP | None = None
_chunks: list[str] = []

# text-embedding-3-large has stronger multilingual (including Arabic) representation than -small
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSION = 3072
INDEX_PATH = "labor_law.index"
CHUNKS_PATH = "labor_law_chunks.pkl"


def _embed(texts: list[str]) -> np.ndarray:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Batch in groups of 100 to stay within API limits
    all_vectors = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_vectors.extend([item.embedding for item in response.data])
    vectors = np.array(all_vectors, dtype="float32")
    # Normalize to unit length so IndexFlatIP == cosine similarity
    faiss.normalize_L2(vectors)
    return vectors


def embed_and_store(chunks: list[str]) -> None:
    global _index, _chunks
    vectors = _embed(chunks)
    _index = faiss.IndexFlatIP(DIMENSION)
    _index.add(vectors)
    _chunks = chunks


def save_to_disk() -> None:
    assert _index is not None, "No index to save"
    faiss.write_index(_index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(_chunks, f)


def load_from_disk() -> bool:
    global _index, _chunks
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return False
    _index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        _chunks = pickle.load(f)
    return True


def is_loaded() -> bool:
    return _index is not None and _index.ntotal > 0


def retrieve_with_scores(query: str, k: int = 5) -> list[tuple[str, float]]:
    if _index is None or _index.ntotal == 0:
        return []
    vector = np.array([_embed([query])[0]], dtype="float32")
    # _embed already normalizes, but ensure single-vector query is also normalized
    faiss.normalize_L2(vector)
    k = min(k, _index.ntotal)
    scores, indices = _index.search(vector, k)
    return [
        (_chunks[idx], float(scores[0][rank]))
        for rank, idx in enumerate(indices[0])
        if idx < len(_chunks)
    ]


def retrieve(query: str, k: int = 5) -> list[str]:
    return [chunk for chunk, _ in retrieve_with_scores(query, k)]
