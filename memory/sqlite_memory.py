"""SQLite adapter for durable chat messages."""

import sqlite3
from pathlib import Path


class SQLiteMemory:
    """Persist and retrieve ordered chat messages for named sessions."""

    def __init__(self, database_path: Path, memory_turns: int) -> None:
        """Configure the database location and maximum conversational context."""
        self._database_path = database_path
        self._memory_turns = memory_turns

    def initialize(self) -> None:
        """Create the messages table when it does not already exist."""
        self._database_path.parent.mkdir(exist_ok=True)
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save(self, session: str, role: str, content: str) -> None:
        """Store one user or assistant message in a named session."""
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                "INSERT INTO messages (session, role, content) VALUES (?, ?, ?)",
                (session, role, content),
            )

    def recent(self, session: str) -> list[tuple[str, str]]:
        """Return the most recent configured number of complete conversation turns."""
        with sqlite3.connect(self._database_path) as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM messages
                WHERE session = ? ORDER BY id DESC LIMIT ?
                """,
                (session, self._memory_turns * 2),
            ).fetchall()
        rows.reverse()
        return [(str(role), str(content)) for role, content in rows]
