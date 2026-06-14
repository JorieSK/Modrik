# Extracts text from uploaded .txt and .pdf files, then splits it into overlapping chunks.
import pdfplumber


def extract_text(file) -> str:
    name = file.name.lower()

    if name.endswith(".txt"):
        raw = file.read().decode("utf-8")
    elif name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            raw = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        raise ValueError(f"Unsupported file type: {file.name}")

    return _chunk(raw)


def _chunk(text: str, size: int = 500, overlap: int = 50) -> str:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return "\n\n---\n\n".join(chunks)
