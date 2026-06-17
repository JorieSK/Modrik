import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_index: faiss.IndexFlatIP | None = None
_chunks: list[str] = []

DIMENSION = 1024
INDEX_PATH = "index/labor_law.index"
CHUNKS_PATH = "index/labor_law_chunks.pkl"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/multilingual-e5-large")
    return _model


def _embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    model = _get_model()
    prefix = "query: " if is_query else "passage: "
    prefixed = [prefix + t for t in texts]
    vectors = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype("float32")


def embed_and_store(chunks: list[str]) -> None:
    global _index, _chunks
    vectors = _embed(chunks, is_query=False)
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
    vector = _embed([query], is_query=True)
    k = min(k, _index.ntotal)
    scores, indices = _index.search(vector, k)
    return [
        (_chunks[idx], float(scores[0][rank]))
        for rank, idx in enumerate(indices[0])
        if idx < len(_chunks)
    ]


def retrieve(query: str, k: int = 5) -> list[str]:
    return [chunk for chunk, _ in retrieve_with_scores(query, k)]
