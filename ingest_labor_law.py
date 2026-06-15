"""
Run once to embed the labor law Excel files into a FAISS index saved to disk.
Usage: python3 ingest_labor_law.py
"""
import pandas as pd
from vector_store import embed_and_store, save_to_disk

DATA_DIR = "data"
# label → (path, status_tag)
# Repealed articles are included so the AI can tell users the article is no longer in force
EXCEL_FILES = {
    "active":   (f"{DATA_DIR}/labor_law_active.xlsx",   None),
    "amended":  (f"{DATA_DIR}/labor_law_amended.xlsx",  "⚠️ هذه المادة معدّلة"),
    "repealed": (f"{DATA_DIR}/labor_law_repealed.xlsx", "🚫 هذه المادة ملغاة ولا تُطبَّق حالياً"),
}
CHUNK_MAX = 600  # chars — articles longer than this get split into sub-chunks


def split_article(header: str, text: str, kw_suffix: str) -> list[str]:
    """
    Split a long article into sub-chunks.
    Prefers splitting on definition markers (-term: ...) then on paragraph breaks.
    Each sub-chunk keeps the article header so retrieval context is clear.
    """
    # Article 2-style: definitions separated by newline + dash
    if "\n\n-" in text:
        raw_parts = text.split("\n\n-")
        parts = [raw_parts[0].strip()]
        parts += [f"-{p.strip()}" for p in raw_parts[1:]]
    else:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Merge consecutive short parts so we don't create too-tiny chunks
    merged: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + "\n\n" + part).strip() if current else part
        if len(candidate) <= CHUNK_MAX:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = part
    if current:
        merged.append(current)

    if not merged:
        return [text]

    return [
        f"{header} (جزء {i})\n{sub}{kw_suffix}"
        for i, sub in enumerate(merged, 1)
    ]


def build_chunks(df: pd.DataFrame, status_tag: str | None = None) -> list[str]:
    chunks = []
    for _, row in df.iterrows():
        article_num = str(row.get("رقم المادة (نصي)", "")).strip()
        chapter     = str(row.get("الفصل / الموضوع", "")).strip()
        text        = str(row.get("نص المادة", "")).strip()
        keywords    = str(row.get("الموضوعات (كلمات مفتاحية)", "")).strip()

        if not text or text == "nan":
            continue

        status_prefix = f"{status_tag}\n" if status_tag else ""
        header    = f"{status_prefix}{article_num} — {chapter}"
        kw_suffix = f"\nالكلمات المفتاحية: {keywords}" if keywords and keywords != "nan" else ""

        if len(text) > CHUNK_MAX:
            chunks.extend(split_article(header, text, kw_suffix))
        else:
            chunks.append(f"{header}\n{text}{kw_suffix}")

    return chunks


def main():
    print("Reading Excel files…")
    all_chunks: list[str] = []
    for label, (path, status_tag) in EXCEL_FILES.items():
        df_part = pd.read_excel(path)
        print(f"  {path}: {len(df_part)} rows ({label})")
        chunks = build_chunks(df_part, status_tag)
        print(f"    >> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"  {len(all_chunks)} total chunks after smart splitting")

    print("Embedding with text-embedding-3-large (multilingual)…")
    embed_and_store(all_chunks)

    print("Saving index to disk…")
    save_to_disk()

    print("Done. labor_law.index and labor_law_chunks.pkl are ready.")


if __name__ == "__main__":
    main()
