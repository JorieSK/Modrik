"""
Pipeline tests — covers every public function in the project.

Modules tested:
  - core.file_service._chunk
  - core.file_service.extract_text (.txt, .pdf, unsupported)
  - core.ai_service.stream_response
  - core.vector_store.embed_and_store / retrieve
  - core.pii_guard.redact_pii
"""

import io
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═══════════════════════════════════════════════════════════════
# 1. core.file_service._chunk
# ═══════════════════════════════════════════════════════════════
class TestChunk(unittest.TestCase):

    def setUp(self):
        from core.file_service import _chunk
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
        self.assertGreater(len(parts), 1)

    def test_empty_string(self):
        result = self.chunk("", size=500, overlap=50)
        self.assertEqual(result, "")


# ═══════════════════════════════════════════════════════════════
# 2. core.file_service.extract_text – .txt path
# ═══════════════════════════════════════════════════════════════
class TestExtractTxt(unittest.TestCase):

    def setUp(self):
        from core.file_service import extract_text
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
        content = ("word " * 300).encode("utf-8")  # ~1500 chars → multiple chunks
        f = self._make_file("long.txt", content)
        result = self.extract_text(f)
        self.assertIn("---", result)

    def test_unsupported_type_raises(self):
        f = self._make_file("doc.docx", b"binary data")
        with self.assertRaises(ValueError):
            self.extract_text(f)


# ═══════════════════════════════════════════════════════════════
# 3. core.file_service.extract_text – .pdf path (mocked via fitz)
# ═══════════════════════════════════════════════════════════════
class TestExtractPdf(unittest.TestCase):

    def setUp(self):
        from core.file_service import extract_text
        self.extract_text = extract_text

    def test_pdf_extraction(self):
        fake_page = MagicMock()
        fake_page.get_text.return_value = "PDF page text here."

        fake_doc = MagicMock()
        fake_doc.__iter__ = MagicMock(return_value=iter([fake_page]))

        f = io.BytesIO(b"%PDF-1.4 fake")
        f.name = "document.pdf"

        with patch("core.file_service.fitz.open", return_value=fake_doc):
            result = self.extract_text(f)

        self.assertIn("PDF page text here.", result)

    def test_pdf_empty_pages(self):
        fake_page = MagicMock()
        fake_page.get_text.return_value = ""  # fitz returns "" for blank pages

        fake_doc = MagicMock()
        fake_doc.__iter__ = MagicMock(return_value=iter([fake_page]))

        f = io.BytesIO(b"%PDF-1.4 fake")
        f.name = "empty.pdf"

        with patch("core.file_service.fitz.open", return_value=fake_doc):
            result = self.extract_text(f)

        self.assertEqual(result, "")


# ═══════════════════════════════════════════════════════════════
# 4. core.ai_service.stream_response
# ═══════════════════════════════════════════════════════════════
class TestStreamResponse(unittest.TestCase):

    def _make_chunk(self, text):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text
        return chunk

    def test_yields_content_tokens(self):
        from core.ai_service import stream_response

        fake_chunks = [self._make_chunk("Hello"), self._make_chunk(" world")]

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(fake_chunks))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("core.ai_service.OpenAI", return_value=mock_client):
            tokens = list(stream_response("hi"))

        self.assertEqual(tokens, ["Hello", " world"])

    def test_none_content_skipped(self):
        """Chunks where delta.content is None must not be yielded."""
        from core.ai_service import stream_response

        chunk_none = MagicMock()
        chunk_none.choices[0].delta.content = None
        chunk_real = self._make_chunk("data")

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([chunk_none, chunk_real]))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("core.ai_service.OpenAI", return_value=mock_client):
            tokens = list(stream_response("hi"))

        self.assertEqual(tokens, ["data"])

    def test_context_becomes_second_system_message(self):
        """Context must appear as a second system message, after SYSTEM_PROMPT."""
        from core.ai_service import stream_response

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([self._make_chunk("ok")]))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("core.ai_service.OpenAI", return_value=mock_client):
            list(stream_response("hello", context="some law context"))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("some law context", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[2]["content"], "hello")

    def test_no_context_sends_system_and_user(self):
        """Without context, exactly system prompt + user message are sent."""
        from core.ai_service import stream_response

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([self._make_chunk("ok")]))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("core.ai_service.OpenAI", return_value=mock_client):
            list(stream_response("hello"))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "hello")

    def test_stream_flag_is_true(self):
        """The OpenAI call must always use stream=True."""
        from core.ai_service import stream_response

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([]))

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        with patch("core.ai_service.OpenAI", return_value=mock_client):
            list(stream_response("test"))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertTrue(call_kwargs.get("stream"), "stream must be True")


# ═══════════════════════════════════════════════════════════════
# 5. core.vector_store.embed_and_store + retrieve
# ═══════════════════════════════════════════════════════════════
class TestVectorStore(unittest.TestCase):

    def _fake_embed(self, texts, is_query=False):
        """Returns deterministic unit vectors without loading the real model."""
        import numpy as np
        rng = np.random.default_rng(seed=42)
        vecs = rng.random((len(texts), 1024)).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_embed_and_store_then_retrieve_returns_strings(self):
        import core.vector_store as vs

        chunks = ["cats are mammals", "dogs bark loudly", "fish live in water"]

        with patch("core.vector_store._embed", side_effect=self._fake_embed):
            vs.embed_and_store(chunks)
            results = vs.retrieve("mammals", k=2)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, str) for r in results))
        self.assertTrue(all(r in chunks for r in results))

    def test_retrieve_k_capped_to_index_size(self):
        import core.vector_store as vs

        chunks = ["only one chunk"]

        with patch("core.vector_store._embed", side_effect=self._fake_embed):
            vs.embed_and_store(chunks)
            results = vs.retrieve("query", k=10)

        self.assertEqual(len(results), 1)

    def test_retrieve_before_store_returns_empty(self):
        import core.vector_store as vs
        vs._index = None
        vs._chunks = []

        results = vs.retrieve("anything")
        self.assertEqual(results, [])

    def test_embed_and_store_resets_previous_index(self):
        import core.vector_store as vs

        with patch("core.vector_store._embed", side_effect=self._fake_embed):
            vs.embed_and_store(["first batch"])
            vs.embed_and_store(["second batch A", "second batch B"])
            results = vs.retrieve("query", k=5)

        self.assertEqual(len(results), 2)
        self.assertTrue(all("second" in r for r in results))


# ═══════════════════════════════════════════════════════════════
# 6. core.pii_guard.redact_pii
# ═══════════════════════════════════════════════════════════════
class TestPiiGuard(unittest.TestCase):

    def setUp(self):
        from core.pii_guard import redact_pii
        self.redact_pii = redact_pii

    def test_saudi_id_redacted(self):
        text = "رقم الهوية: 1234567890"
        result, count = self.redact_pii(text)
        self.assertNotIn("1234567890", result)
        self.assertIn("[رقم هوية]", result)
        self.assertEqual(count, 1)

    def test_iqama_id_redacted(self):
        text = "رقم الإقامة: 2987654321"
        result, count = self.redact_pii(text)
        self.assertNotIn("2987654321", result)
        self.assertIn("[رقم هوية]", result)
        self.assertEqual(count, 1)

    def test_iban_redacted(self):
        text = "رقم IBAN: SA1234567890123456789012"
        result, count = self.redact_pii(text)
        self.assertNotIn("SA1234567890123456789012", result)
        self.assertIn("[رقم IBAN]", result)
        self.assertEqual(count, 1)

    def test_no_pii_returns_unchanged(self):
        text = "هذا نص عادي بدون بيانات حساسة"
        result, count = self.redact_pii(text)
        self.assertEqual(result, text)
        self.assertEqual(count, 0)

    def test_multiple_pii_all_redacted(self):
        text = "هوية: 1234567890 وIBAN: SA0987654321098765432109"
        result, count = self.redact_pii(text)
        self.assertEqual(count, 2)
        self.assertNotIn("1234567890", result)
        self.assertNotIn("SA0987654321098765432109", result)

    def test_id_inside_longer_number_not_matched(self):
        """An 11-digit number must not be treated as a Saudi ID."""
        text = "الرقم: 12345678901"
        result, count = self.redact_pii(text)
        self.assertEqual(count, 0)
        self.assertEqual(result, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
