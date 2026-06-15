# Extracts text from uploaded .txt and .pdf files.
# Uses PyMuPDF (fitz) for PDFs — it handles Arabic RTL glyph ordering correctly,
# unlike pdfplumber which reverses Arabic character order in many PDF encodings.
import fitz
import io


def extract_text(file) -> str:
    name = file.name.lower()

    if name.endswith(".txt"):
        return file.read().decode("utf-8")

    if name.endswith(".pdf"):
        data = file.read()
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(p for p in pages if p.strip())

    raise ValueError(f"Unsupported file type: {file.name}")
