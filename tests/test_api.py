"""Tests for API response formatting without starting models or a web server."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from api import (
    ChatRequest,
    _delete_uploaded_document,
    _list_uploaded_documents,
    _load_and_chunk_uploads,
    _sources_for_document_ids,
    _sse_event,
)
from config import get_settings


class ApiTests(unittest.TestCase):
    """Verify request validation and Server-Sent Event formatting."""

    def test_chat_request_rejects_invalid_mmr_lambda(self) -> None:
        """The public API enforces the same MMR range as the CLI."""
        with self.assertRaises(ValueError):
            ChatRequest(question="What is RAG?", mmr_lambda=1.1)

    def test_sse_event_encodes_a_json_payload(self) -> None:
        """Token text is safely JSON encoded into a valid SSE event."""
        self.assertEqual(_sse_event("token", {"text": "RAG\nworks"}), 'event: token\ndata: {"text": "RAG\\nworks"}\n\n')

    def test_uploaded_documents_receive_their_hr_category(self) -> None:
        """Uploaded chunks preserve the category used to scope later questions."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "candidate.txt"
            path.write_text("Python and SQL experience.", encoding="utf-8")
            chunks = _load_and_chunk_uploads([path], "resume", get_settings())

        self.assertEqual(chunks[0].metadata["document_type"], "resume")

    def test_uploaded_documents_keep_their_public_selection_metadata(self) -> None:
        """New uploads retain the ID and name used by the document picker."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "certificate.txt"
            path.write_text("Certificate text.", encoding="utf-8")
            chunks = _load_and_chunk_uploads(
                [path],
                "other",
                get_settings(),
                {path: {"document_id": "certificate-id", "document_name": "certificate.txt"}},
            )

        self.assertEqual(chunks[0].metadata["document_id"], "certificate-id")
        self.assertEqual(chunks[0].metadata["document_name"], "certificate.txt")

    def test_document_ids_resolve_to_only_the_selected_source(self) -> None:
        """The API maps a public document ID to its internal Chroma source path."""
        vectorstore = Mock()
        vectorstore.get.return_value = {
            "metadatas": [
                {
                    "source": "docs/uploaded/other/internal-id_certificate.pdf",
                    "document_id": "certificate-id",
                    "document_name": "certificate.pdf",
                }
            ]
        }

        documents = _list_uploaded_documents(vectorstore)

        self.assertEqual(documents[0].id, "certificate-id")
        self.assertEqual(documents[0].name, "certificate.pdf")
        self.assertEqual(
            _sources_for_document_ids(vectorstore, ["certificate-id"]),
            ["docs/uploaded/other/internal-id_certificate.pdf"],
        )

    def test_delete_uploaded_document_removes_its_file_and_index_chunks(self) -> None:
        """Deleting an upload removes its persisted source and its Chroma records."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            docs_dir = Path(temporary_directory) / "docs"
            document_id = "a" * 32
            uploaded_file = docs_dir / "uploaded" / "other" / f"{document_id}_certificate.txt"
            uploaded_file.parent.mkdir(parents=True)
            uploaded_file.write_text("Certificate text.", encoding="utf-8")
            vectorstore = Mock()
            vectorstore.get.return_value = {
                "metadatas": [
                    {
                        "source": str(uploaded_file),
                        "document_id": document_id,
                        "document_name": "certificate.txt",
                    }
                ]
            }

            deleted = _delete_uploaded_document(vectorstore, docs_dir, document_id)
            self.assertFalse(uploaded_file.exists())

        self.assertEqual(deleted.id, document_id)
        self.assertEqual(deleted.name, "certificate.txt")
        vectorstore.delete.assert_called_once_with(where={"source": str(uploaded_file)})
