import fitz
import io

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 100


def _chunk(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> str:
    """Split long text into overlapping chunks separated by '---'."""
    if not text or len(text) <= size:
        return text
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return "\n\n---\n\n".join(chunks)


def extract_text(file) -> str:
    name = file.name.lower()

    if name.endswith(".txt"):
        raw = file.read().decode("utf-8")
        return _chunk(raw)

    if name.endswith(".pdf"):
        data = file.read()
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(p for p in pages if p.strip())

    raise ValueError(f"Unsupported file type: {file.name}")
