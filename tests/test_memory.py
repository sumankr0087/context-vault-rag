"""Tests for SQLite chat-memory ordering and limits."""

import tempfile
import unittest
from os import environ
from pathlib import Path
from unittest.mock import patch

from config import get_settings
from llm.ollama_client import create_chat_model
from memory.sqlite_memory import SQLiteMemory


class SQLiteMemoryTests(unittest.TestCase):
    """Verify durable conversation memory's public contract."""

    def test_recent_returns_oldest_to_newest_within_limit(self) -> None:
        """The prompt receives chronological messages from only recent turns."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            memory = SQLiteMemory(Path(temporary_directory) / "memory.db", memory_turns=1)
            memory.initialize()
            for role, content in [("user", "first"), ("assistant", "one"), ("user", "second")]:
                memory.save("demo", role, content)
            self.assertEqual(memory.recent("demo"), [("assistant", "one"), ("user", "second")])

    def test_embedding_context_has_a_safe_default(self) -> None:
        """The local embedding backend receives the verified conservative context window."""
        with patch.dict(environ, {}, clear=True):
            self.assertEqual(get_settings().embedding_num_ctx, 1024)

    def test_chat_model_disables_reasoning_output(self) -> None:
        """The CLI asks Ollama for final answers rather than its thinking trace."""
        self.assertFalse(create_chat_model(get_settings()).reasoning)
