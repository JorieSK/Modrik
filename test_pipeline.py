"""
Pipeline test: verifies every method added in the last push.
  - file_service._chunk()
  - file_service.extract_text() for .txt
  - file_service.extract_text() for .pdf
  - file_service.extract_text() for unsupported type (error path)
  - ai_service.stream_response() (mocked – no real API key needed)
  - vector_store.embed_and_store()
  - vector_store.retrieve()
"""

import io
import sys
import os
import types
import unittest
from unittest.mock import patch, MagicMock

# ── make sure the project root is on the path ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))


# ═══════════════════════════════════════════════════════════════
# 1. file_service._chunk
# ═══════════════════════════════════════════════════════════════
class TestChunk(unittest.TestCase):

    def setUp(self):
        from file_service import _chunk
        self.chunk = _chunk

    def test_short_text_single_chunk(self):
        text = "Hello world"
        result = self.chunk(text, size=500, overlap=50)
        self.assertEqual(result, "Hello world")

    def test_multiple_chunks_separator(self):
        text = "A" * 600
        result = self.chunk(text, size=500, overlap=50)
        self.assertIn("---", result, "Chunks must be separated by ---")
        parts = result.split("\n\n---\n\n")
        self.assertEqual(len(parts), 2)

    def test_overlap_produces_shared_content(self):
        text = "X" * 100
        result = self.chunk(text, size=60, overlap=20)
        parts = result.split("\n\n---\n\n")
        # second chunk starts 40 chars in, so total coverage exceeds 100 without overlap
        self.assertGreater(len(parts), 1)

    def test_empty_string(self):
        result = self.chunk("", size=500, overlap=50)
        self.assertEqual(result, "")


# ═══════════════════════════════════════════════════════════════
# 2. file_service.extract_text – .txt path
# ═══════════════════════════════════════════════════════════════
class TestExtractTxt(unittest.TestCase):

    def setUp(self):
        from file_service import extract_text
        self.extract_text = extract_text

    def _make_file(self, name: str, content: bytes):
        f = io.BytesIO(content)
        f.name = name
        return f

    def test_txt_extraction(self):
        content = b"Hello from a text file."
        f = self._make_file("sample.txt", content)
        result = self.extract_text(f)
        self.assertIn("Hello from a text file.", result)

    def test_txt_long_content_is_chunked(self):
        content = ("word " * 300).encode("utf-8")   # ~1500 chars → multiple chunks
        f = self._make_file("long.txt", content)
        result = self.extract_text(f)
        self.assertIn("---", result)

    def test_unsupported_type_raises(self):
        f = self._make_file("doc.docx", b"binary data")
        with self.assertRaises(ValueError):
            self.extract_text(f)


# ═══════════════════════════════════════════════════════════════
# 3. file_service.extract_text – .pdf path
# ═══════════════════════════════════════════════════════════════
class TestExtractPdf(unittest.TestCase):

    def setUp(self):
        from file_service import extract_text
        self.extract_text = extract_text

    def test_pdf_extraction(self):
        """Mock pdfplumber so we don't need a real PDF file."""
        fake_page = MagicMock()
        fake_page.extract_text.return_value = "PDF page text here."

        fake_pdf = MagicMock()
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)
        fake_pdf.pages = [fake_page]

        f = io.BytesIO(b"%PDF-1.4 fake")
        f.name = "document.pdf"

        with patch("file_service.pdfplumber.open", return_value=fake_pdf):
            result = self.extract_text(f)

        self.assertIn("PDF page text here.", result)

    def test_pdf_empty_pages(self):
        """Pages that return None from extract_text should produce empty string."""
        fake_page = MagicMock()
        fake_page.extract_text.return_value = None

        fake_pdf = MagicMock()
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)
        fake_pdf.pages = [fake_page]

        f = io.BytesIO(b"%PDF-1.4 fake")
        f.name = "empty.pdf"

        with patch("file_service.pdfplumber.open", return_value=fake_pdf):
            result = self.extract_text(f)

        self.assertEqual(result, "")


# ═══════════════════════════════════════════════════════════════
# 4. ai_service.stream_response
# ═══════════════════════════════════════════════════════════════
class TestStreamResponse(unittest.TestCase):

    def _make_chunk(self, text):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text
        return chunk

    def test_yields_content_tokens(self):
        from ai_service import stream_response

        fake_chunks = [self._make_chunk("Hello"), self._make_chunk(" world")]

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(fake_chunks))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("ai_service.OpenAI", return_value=mock_client):
            tokens = list(stream_response("hi"))

        self.assertEqual(tokens, ["Hello", " world"])

    def test_none_content_skipped(self):
        """Chunks where delta.content is None must not be yielded."""
        from ai_service import stream_response

        chunk_none = MagicMock()
        chunk_none.choices[0].delta.content = None
        chunk_real = self._make_chunk("data")

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([chunk_none, chunk_real]))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("ai_service.OpenAI", return_value=mock_client):
            tokens = list(stream_response("hi"))

        self.assertEqual(tokens, ["data"])

    def test_context_becomes_system_message(self):
        """When context is provided it must appear as a system message."""
        from ai_service import stream_response

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([self._make_chunk("ok")]))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("ai_service.OpenAI", return_value=mock_client):
            list(stream_response("hello", context="You are a pirate."))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a pirate.")
        self.assertEqual(messages[1]["role"], "user")

    def test_no_context_no_system_message(self):
        """Without context, only the user message should be sent."""
        from ai_service import stream_response

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([self._make_chunk("ok")]))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("ai_service.OpenAI", return_value=mock_client):
            list(stream_response("hello"))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")

    def test_stream_flag_is_true(self):
        """The OpenAI call must always use stream=True."""
        from ai_service import stream_response

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([]))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("ai_service.OpenAI", return_value=mock_client):
            list(stream_response("test"))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertTrue(call_kwargs.get("stream"), "stream must be True")


# ═══════════════════════════════════════════════════════════════
# 5. vector_store.embed_and_store + retrieve
# ═══════════════════════════════════════════════════════════════
class TestVectorStore(unittest.TestCase):

    def _fake_embed(self, texts):
        """Returns deterministic unit vectors so we don't need a real API key."""
        import numpy as np
        rng = np.random.default_rng(seed=42)
        vecs = rng.random((len(texts), 1536)).astype("float32")
        # Normalise so cosine ~ L2
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def _make_embedding_response(self, texts):
        response = MagicMock()
        vectors = self._fake_embed(texts)
        response.data = [
            MagicMock(embedding=vectors[i].tolist()) for i in range(len(texts))
        ]
        return response

    def test_embed_and_store_then_retrieve_returns_strings(self):
        import vector_store

        chunks = ["cats are mammals", "dogs bark loudly", "fish live in water"]

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda **kw: \
            self._make_embedding_response(kw["input"])

        with patch("vector_store.OpenAI", return_value=mock_client):
            vector_store.embed_and_store(chunks)
            results = vector_store.retrieve("mammals", k=2)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, str) for r in results))
        self.assertTrue(all(r in chunks for r in results))

    def test_retrieve_k_capped_to_index_size(self):
        import vector_store

        chunks = ["only one chunk"]

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda **kw: \
            self._make_embedding_response(kw["input"])

        with patch("vector_store.OpenAI", return_value=mock_client):
            vector_store.embed_and_store(chunks)
            results = vector_store.retrieve("query", k=10)

        self.assertEqual(len(results), 1)

    def test_retrieve_before_store_returns_empty(self):
        import vector_store
        vector_store._index = None
        vector_store._chunks = []

        results = vector_store.retrieve("anything")
        self.assertEqual(results, [])

    def test_embed_and_store_resets_previous_index(self):
        import vector_store

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda **kw: \
            self._make_embedding_response(kw["input"])

        with patch("vector_store.OpenAI", return_value=mock_client):
            vector_store.embed_and_store(["first batch"])
            vector_store.embed_and_store(["second batch A", "second batch B"])
            results = vector_store.retrieve("query", k=5)

        # After re-indexing only 2 chunks should exist
        self.assertEqual(len(results), 2)
        self.assertTrue(all("second" in r for r in results))


if __name__ == "__main__":
    unittest.main(verbosity=2)
